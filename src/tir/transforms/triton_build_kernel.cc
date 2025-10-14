/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/name_supply.h>
#include <tvm/support/iterator.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../target/source/codegen_triton.h"
#include "../analysis/var_use_def_analysis.h"
#include "tvm/runtime/object.h"

namespace tvm::tir {

namespace {

struct DeviceFuncInfo {
  GlobalVar gvar;
  PrimFunc func;
  Map<Var, Buffer> buffer_map;
  Array<ObjectRef> tir_params;
};

struct TritonSourceKernelInfo {
  String kernel_name;
  String kernel_src;
  Array<PrimExpr> grid_shape;
  Integer num_warps;
  Integer num_stages;
};

TritonSourceKernelInfo TVMFuncToTritonSource(const String& func_name,
                                             const DeviceFuncInfo& dev_func);

class HostDeviceSplitter : public StmtMutator {
 public:
  static PrimFunc Split(const std::string& kernel_name_prefix, PrimFunc func,
                        std::vector<DeviceFuncInfo>& device_funcs) {
    HostDeviceSplitter splitter(kernel_name_prefix);
    func = splitter.VisitFunc_(func);
    device_funcs.insert(device_funcs.end(), splitter.device_funcs_.begin(),
                        splitter.device_funcs_.end());
    return func;
  }

 private:
  explicit HostDeviceSplitter(const std::string& kernel_name_prefix)
      : kernel_name_prefix_(kernel_name_prefix) {}

  PrimFunc VisitFunc_(PrimFunc func);

  Stmt VisitStmt_(const AttrStmtNode* op) final;

  Stmt SplitDeviceFunc(Stmt body, Target device_target);

  const std::string& kernel_name_prefix_;
  GlobalVarSupply var_supply_;
  Map<Var, Integer> current_func_params_;
  std::vector<DeviceFuncInfo> device_funcs_;
};

using GVToKernelMap =
    std::unordered_map<GlobalVar, std::pair<String, Array<PrimExpr>>, ObjectHash, ObjectEqual>;
class KernelUserMutator : public StmtExprMutator {
 public:
  KernelUserMutator(const GVToKernelMap& kernel_calls) : kernel_calls_(kernel_calls) {}

  PrimExpr VisitExpr_(const CallNode* call) override {
    auto* gv = call->op.as<GlobalVarNode>();
    auto it = kernel_calls_.find(GetRef<GlobalVar>(gv));
    if (it == kernel_calls_.end()) {
      return StmtExprMutator::VisitExpr_(call);
    }
    auto [kernel_name, launch_args] = (*it).second;
    // First argument to tvm_call_packed is the kernel name (as registered in the external module),
    // followed by the arguments at the call site. `launch_args` goes at the end.
    Array<PrimExpr> call_args({StringImm(kernel_name)});
    call_args.insert(call_args.end(), call->args.begin(), call->args.end());
    call_args.insert(call_args.end(), launch_args.begin(), launch_args.end());
    return Call(DataType::Void(), builtin::tvm_call_packed(), call_args);
  }

  const GVToKernelMap& kernel_calls_;
};

std::pair<IRModule, std::vector<DeviceFuncInfo>> SplitHostDevice(IRModule mod) {
  static const TargetKind cuda_target = TargetKind::Get("cuda").value(),
                          rocm_target = TargetKind::Get("rocm").value();

  std::vector<DeviceFuncInfo> device_funcs;
  for (auto [gvar, func] : mod->functions) {
    // Actively check if `func` is a PrimFunc. If any function in `mod` is not a PrimFunc,
    // this will throw an error.
    auto pfunc = Downcast<PrimFunc>(func);
    // Only apply to CUDA-targeted functions with `tir.tile_expr_form` attribute.
    auto target = pfunc->GetAttr<Target>("target");
    ICHECK(target.defined()) << "Expected target attribute on function; run BindTarget first";
    Bool is_tile_expr_form = pfunc->GetAttr<Bool>("tir.tile_expr_form", Bool(false)).value();
    auto kind = target.value()->kind;
    if ((kind != cuda_target && kind != rocm_target) || !is_tile_expr_form) {
      continue;
    }
    auto new_host_func =
        HostDeviceSplitter::Split(gvar->name_hint + "_kernel", pfunc, device_funcs);
    if (!new_host_func.same_as(pfunc)) {
      mod->Update(gvar, new_host_func);
    }
  }
  return std::make_pair(mod, device_funcs);
}

auto FindAndCheckFunc(const std::string& name) {
  auto* func = runtime::Registry::Get(name);
  ICHECK(func) << name << " is not registered.";
  return (*func);
}

}  // namespace

IRModule TritonBuildKernelPass(IRModule mod) {
  // 1. Select the functions we want to translate into Triton kernels, and apply
  // our custom version of SplitHostDevice (a PrimFunc -> Array<PrimFunc> version).
  // Each function is converted into a host function and updated in `mod_`, while multiple
  // device functions are created and placed in `device_funcs`.
  auto [mod_, device_funcs] = SplitHostDevice(mod);

  // 2. Build Triton kernels for all functions in `device_funcs`.
  // We'll translate each triton-ready function into an external module, so a call to that function
  // needs to be with a `tvm_call_packed` call. We record this mapping in `kernel_calls`.
  static TypedPackedFunc<Array<ObjectRef>(String, String, Array<PrimExpr>, Array<ObjectRef>,
                                          Optional<Integer>, Optional<Integer>)>
      triton_str_to_device_module_ =
          FindAndCheckFunc("runtime.triton.compile_triton_source_to_device_module");
  Array<runtime::Module> external_mods =
      mod_->attrs.GetAttr<Array<runtime::Module>>("external_mods").value_or({});
  GVToKernelMap kernel_calls;
  for (auto device_func : device_funcs) {
    auto gvar = device_func.gvar;
    // Get Triton source string for the kernel.
    auto triton_src = TVMFuncToTritonSource(gvar->name_hint, device_func);
    // Apply `compile_triton_source_to_device_module`, a Triton-source-to-PTX codegen function
    // registered in TVM Python.
    Array<ObjectRef> results = triton_str_to_device_module_(
        triton_src.kernel_src, triton_src.kernel_name, triton_src.grid_shape,
        device_func.tir_params, triton_src.num_warps, triton_src.num_stages);
    ICHECK_EQ(results.size(), 3);
    // `kernel_name` might differ from `triton_src.kernel_name` if the compiler decides to.
    auto kernel_name = Downcast<String>(results[0]);
    auto ptx_module = Downcast<runtime::Module>(results[1]);
    auto tvm_launch_args = Downcast<Array<PrimExpr>>(results[2]);
    external_mods.push_back(ptx_module);
    kernel_calls.emplace(gvar, std::make_pair(kernel_name, tvm_launch_args));
  }

  // 3. Apply the `kernel_calls` mapping to `mod_`.
  KernelUserMutator mutator(kernel_calls);
  for (auto [gvar, func] : mod_->functions) {
    auto pfunc = Downcast<PrimFunc>(func);
    Stmt new_body = mutator(pfunc->body);
    if (!new_body.same_as(pfunc->body)) {
      pfunc.CopyOnWrite()->body = std::move(new_body);
      mod_->Update(gvar, pfunc);
    }
  }

  // 4. Add the external modules to the module's attributes and return.
  Map<String, ObjectRef> attrs = mod_->attrs->dict;
  attrs.Set("external_mods", external_mods);
  return WithAttrs(mod_, attrs);
}

//! \brief A patched version of TritonBuildKernelPass that generates Triton kernels as strings and
//! returns them, without further incorporating them into the IRModule.
Map<String, Array<ObjectRef>> TritonCollectKernel(IRModule mod) {
  Map<String, Array<ObjectRef>> result;
  auto [mod_, device_funcs] = SplitHostDevice(mod);
  GVToKernelMap kernel_calls;
  for (auto dev_func : device_funcs) {
    auto gvar = dev_func.gvar;
    auto triton_src = TVMFuncToTritonSource(gvar->name_hint, dev_func);
    result.Set(triton_src.kernel_name, {triton_src.kernel_src, triton_src.num_warps,
                                        triton_src.num_stages, triton_src.grid_shape});
  }
  return result;
}

namespace {

PrimFunc HostDeviceSplitter::VisitFunc_(PrimFunc func) {
  current_func_params_.clear();
  for (size_t i = 0; i < func->params.size(); ++i) {
    auto param = func->params[i];
    auto it = func->buffer_map.find(param);
    if (it != func->buffer_map.end()) {
      param = (*it).second->data;
    }
    current_func_params_.Set(param, Integer(i));
  }
  if (auto body = VisitStmt(func->body); !body.same_as(func->body)) {
    func.CopyOnWrite()->body = body;
  }
  return func;
}

Stmt HostDeviceSplitter::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tvm::attr::kTarget) {
    auto device_target = op->node.as<Target>().value().WithoutHost();
    return SplitDeviceFunc(op->body, device_target);
  }
  return StmtMutator::VisitStmt_(op);
}

Stmt HostDeviceSplitter::SplitDeviceFunc(Stmt body, Target device_target) {
  auto [params, buffers_to_declare] = [&]() -> std::tuple<Array<Var>, Array<Buffer>> {
    VarUseDefAnalyzer use_def(/*defined_vars=*/{}, /*visit_thread_extent=*/true);
    use_def(body);

    std::vector<Var> params{use_def.undefined_.begin(), use_def.undefined_.end()};
    // If the variable exists in the current function, put them at the front (keep the order);
    // otherwise, sort them by handle type first, then by name.
    auto SortKey = [this](const Var& var) {
      auto it = current_func_params_.find(var);
      if (it != current_func_params_.end()) {
        return std::make_tuple((*it).second->value, false, String());
      } else {
        static const auto max = std::numeric_limits<int64_t>::max();
        return std::make_tuple(max, !var->dtype.is_handle(), var->name_hint);
      }
    };
    std::sort(params.begin(), params.end(),
              [&SortKey](const Var& a, const Var& b) { return SortKey(a) < SortKey(b); });
    return {params, use_def.undefined_buffers_};
  }();

  // CodeGenCPU is used for some device-side targets, such as
  // "ext_dev", and expects to be able to return a int32_t status
  // code.
  bool can_propagate_errors = [&]() {
    auto kind = device_target->GetTargetDeviceType();
    return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
  }();
  IntImm success(DataType::Int(32), 0);
  Type kernel_ret_type;
  if (can_propagate_errors) {
    kernel_ret_type = PrimType(DataType::Int(32));
    body = SeqStmt::Flatten(body, Evaluate(ret(success)));
  } else {
    kernel_ret_type = VoidType();
  }

  for (Buffer buf : buffers_to_declare) {
    body = DeclBuffer(buf, std::move(body));
  }
  auto device_func =
      WithAttrs(PrimFunc(params, body, kernel_ret_type), {{tvm::attr::kTarget, device_target},
                                                          {tir::attr::kNoAlias, Bool(true)},
                                                          {tir::attr::kIsGlobalFunc, Bool(true)}});
  GlobalVar kernel_gv = var_supply_->FreshGlobal(kernel_name_prefix_, /*add_prefix=*/false);
  // Restructure the parameters of the device function as an array of (PrimExpr | Buffer).
  auto buffer_map = support::map(buffers_to_declare, [](const Buffer& buf) {
                      return std::make_pair(buf->data, buf);
                    }).to_container<Map<Var, Buffer>>();
  auto param_with_bufs = support::map(params, [&buffer_map](const Var& var) {
                           auto it = buffer_map.find(var);
                           return it == buffer_map.end() ? (ObjectRef)var : (ObjectRef)(*it).second;
                         }).to_container<Array>();
  device_funcs_.push_back(DeviceFuncInfo{kernel_gv, std::move(device_func), std::move(buffer_map),
                                         std::move(param_with_bufs)});

  // Done with the device function. Now we create a call to the device function in the host
  // function.
  Array<PrimExpr> args = params.Map([](const Var& var) -> PrimExpr { return var; });
  if (can_propagate_errors) {
    Var kernel_error_code("kernel_error_code", success->dtype);
    Call kernel_call(success->dtype, kernel_gv, args);
    AssertStmt assert_success(kernel_error_code == success,
                              StringImm("Error executing compute kernel"), Evaluate(0));
    LetStmt let_check(kernel_error_code, kernel_call, assert_success);
    return std::move(let_check);
  } else {
    return Evaluate(Call(DataType::Void(), kernel_gv, args));
  }
}

struct MakeBlockPtrArgs {
  Buffer buffer;
  PrimExpr offset;
  Array<PrimExpr> strides, shape, offsets, block_shape, order;

  Array<PrimExpr> PackArgs() {
    Array<PrimExpr> args({buffer->data, offset, Integer(shape.size())});
    args.insert(args.end(), shape.begin(), shape.end());
    args.insert(args.end(), strides.begin(), strides.end());
    args.insert(args.end(), offsets.begin(), offsets.end());
    args.insert(args.end(), block_shape.begin(), block_shape.end());
    args.insert(args.end(), order.begin(), order.end());
    return args;
  }
};

struct BufferDimIndexInfo {
  PrimExpr shape, strides;
  Range index;
  size_t order{0};

  BufferDimIndexInfo(PrimExpr shape, PrimExpr strides, Range index)
      : shape(shape), strides(strides), index(index) {}
};

struct TritonRealizePointerMutator : public StmtExprMutator {
 public:
  static PrimFunc VisitFunc(PrimFunc func, const Map<Var, Buffer>& buffer_map) {
    TritonRealizePointerMutator mutator(buffer_map);
    auto new_stmt = mutator(func->body);
    // To keep the AST structure simpler, we put the stores of range buffers right before
    // `new_stmt`, and declare the buffers right before that.
    Array<Stmt> stmts = support::map(mutator.range_buffers_, [](const auto& pair) {
                          return (Stmt)pair.second;
                        }).to_container<Array>();
    stmts.push_back(new_stmt);
    new_stmt = SeqStmt::Flatten(stmts);
    for (auto [_, store] : mutator.range_buffers_) {
      new_stmt = DeclBuffer(store->region->buffer, new_stmt);
    }
    func.CopyOnWrite()->body = new_stmt;
    return func;
  }

 private:
  TritonRealizePointerMutator(const Map<Var, Buffer>& buffer_map) : global_buffers_(buffer_map) {}

  Stmt VisitStmt_(const BlockNode* block) {
    LOG(FATAL) << "Block is not supported; run LowerOpaqueBlock first";
  }

  // Realize `region = value` as `triton_store(region, value)` if `region` is a block pointer.
  Stmt VisitStmt_(const BufferRegionStoreNode* store) {
    // Check if an `all_store_mask_` is defined, and we have a pointer store. If so, extend the
    // store region to the padded region, and store with mask.
    auto padded_store = [this, store]() -> std::optional<std::pair<PrimExpr, PrimExpr>> {
      if (!all_store_mask_.defined() || global_buffers_.count(store->region->buffer->data) == 0) {
        return std::nullopt;
      }
      return CreatePaddedRegionWithMask(store->region, /*n_dims=*/std::nullopt);
    }();
    auto rhs = VisitExpr(store->value);
    if (padded_store) {
      auto [padded_block_ptr, mask] = padded_store.value();
      return Evaluate(
          Call(DataType::Void(), builtin::triton_store(), {padded_block_ptr, rhs, mask}));
    } else if (auto lhs_blk_ptr = MakeTritonBlockPtr(store->region)) {
      return Evaluate(Call(DataType::Void(), builtin::triton_store(), {lhs_blk_ptr.value(), rhs}));
    } else if (!rhs.same_as(store->value)) {
      return BufferRegionStore(store->region, rhs, store->span);
    } else {
      return GetRef<BufferRegionStore>(store);
    }
  }

  PrimExpr VisitExpr_(const BufferRegionNode* region_node) {
    auto region = GetRef<BufferRegion>(region_node);
    if (auto blk_ptr = MakeTritonBlockPtr(region)) {
      return Call(region->buffer->dtype, builtin::triton_load(), {blk_ptr.value()});
    } else {
      return region;
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load_node) {
    // Independent `BufferLoad`s can still show up in the IR (e.g. in loop bounds).
    // We convert it into a 0D BufferRegion.
    auto ranges = support::map(load_node->indices, [](const PrimExpr& index) {
                    return Range::FromMinExtent(index, 1);
                  }).to_container<Array>();
    auto region = BufferRegion(load_node->buffer, ranges, /*output_to_input_dims=*/{});
    if (auto blk_ptr = MakeTritonBlockPtr(region)) {
      return Call(load_node->dtype, builtin::triton_load(), {blk_ptr.value()});
    } else {
      return GetRef<BufferLoad>(load_node);
    }
  }

  PrimExpr VisitExpr_(const CallNode* call) {
    if (call->op.same_as(builtin::tile_dot()) && !all_store_mask_.defined()) {
      if (auto lhs_region = call->args[0].as<BufferRegionNode>()) {
        if (auto padded =
                CreatePaddedRegionWithMask(GetRef<BufferRegion>(lhs_region), /*n_dims=*/2)) {
          auto [lhs_block_ptr, mask] = padded.value();
          auto lhs_ptr_load =
              Call(lhs_region->buffer->dtype, builtin::triton_load(), {lhs_block_ptr, mask});
          all_store_mask_ = mask;
          auto new_args =
              support::map(call->args.begin() + 1, call->args.end(), [this](const PrimExpr& arg) {
                return this->VisitExpr(arg);
              }).to_container<Array>();
          new_args.insert(new_args.begin(), lhs_ptr_load);
          return Call(call->dtype, call->op, new_args);
        }
      }
    } else if (call->op.same_as(builtin::tile_permute())) {
      // Due to how BufferRegionStore is structured, `triton_permute` can only occur in RHS.
      auto region = Downcast<BufferRegion>(call->args[0]);
      Array<Integer> perm;
      for (size_t i = 1; i < call->args.size(); ++i) {
        perm.push_back(Downcast<Integer>(call->args[i]));
      }
      if (auto blk_ptr = MakeTritonBlockPtr(region, perm)) {
        return Call(region->buffer->dtype, builtin::triton_load(), {blk_ptr.value()});
      } else {
        return GetRef<Call>(call);
      }
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  PrimExpr VisitExpr_(const VarNode* var) {
    ICHECK(var->dtype.code() != DataType::kHandle)
        << "Naked buffer use (such as b.data) is invalid; got " << GetRef<PrimExpr>(var);
    return GetRef<PrimExpr>(var);
  }

  Optional<PrimExpr> MakeTritonBlockPtr(const BufferRegion& region,
                                        const Array<Integer>& permutation = {}) {
    auto buffer = region->buffer;
    if (global_buffers_.count(buffer->data) == 0) {
      return NullOpt;
    }
    /* Explanation: we found that exprs of the form `ptr + offset + range(...)[..., None, ...] + ...
       actually uses fewer registers and is faster than using Triton's `tl.make_block_ptr`,
       so we offer that when there is no permutation and no dim unsqueezing. */
    Array<PrimExpr> strides_ = buffer.MakeStrideView()->strides;
    if (permutation.empty()) {
      return MakePointerExpr(region);
    } else {
      return MakeBlockPtrCall(region, permutation);
    }
  }

  PrimExpr MakePointerExpr(const BufferRegion& region) {
    auto buffer = region->buffer;
    auto [offset, dim_infos] = BuildBufferIndexingInfo(buffer, region->ConvertToIndices(), {});
    auto dtype = DataType::Int(32);
    PrimExpr ptr = Call(dtype, builtin::triton_ptr_to_int(), {buffer->data}) + offset;
    for (size_t out_dim = 0; out_dim < dim_infos.size(); ++out_dim) {
      auto dim_info = dim_infos[out_dim];
      auto& range = dim_info.index;
      auto extent = Downcast<Integer>(range->extent);
      if (extent->value == 1) {
        continue;
      }
      PrimExpr stride = dim_info.strides;
      auto region_expr = MakeAndLoadRangeBuffer(extent, dim_infos.size(), out_dim);
      ptr += range->min * stride + region_expr * stride;
    }
    return ptr;
  }

  Call MakeBlockPtrCall(const BufferRegion& region, const Array<Integer>& permutation) {
    auto buffer = region->buffer;
    auto [offset, dim_infos] =
        BuildBufferIndexingInfo(buffer, region->ConvertToIndices(), permutation);
    std::vector<PrimExpr> shape, strides, offsets, block_shape, order;
    for (auto dim : dim_infos) {
      shape.push_back(dim.shape);
      strides.push_back(dim.strides);
      offsets.push_back(dim.index->min);
      block_shape.push_back(dim.index->extent);
      order.push_back(Integer(dim.order));
    }
    MakeBlockPtrArgs margs{buffer, offset, strides, shape, offsets, block_shape, order};
    return Call(DataType::Handle(), builtin::triton_make_block_ptr(), margs.PackArgs());
  }

  std::tuple<PrimExpr, std::vector<BufferDimIndexInfo>> BuildBufferIndexingInfo(
      const Buffer& buffer, const Array<ExprOrRangeOrNull>& indices,
      const Array<Integer>& permutation) {
    Array<PrimExpr> buf_strides = buffer.MakeStrideView()->strides;
    ICHECK_EQ(buf_strides.size(), buffer->shape.size());
    PrimExpr offset = 0;
    size_t dim = 0;
    std::vector<BufferDimIndexInfo> dim_infos;
    for (const auto& index : indices) {
      if (auto expr = index.as<PrimExpr>()) {
        ICHECK(dim < buf_strides.size());
        offset += expr.value() * buf_strides[dim];
        dim++;
      } else if (auto range = index.as<Range>()) {
        ICHECK(dim < buf_strides.size());
        dim_infos.push_back({buffer->shape[dim], buf_strides[dim], range.value()});
        dim++;
      } else {
        ICHECK(dim <= buf_strides.size());  // less than or equal: the last dim can be unsqueezed
        PrimExpr stride = dim < buf_strides.size() ? buf_strides[dim] : 1;
        dim_infos.push_back({Integer(1), stride, Range::FromMinExtent(0, 1)});
        // Don't increment `dim` here.
      }
    }
    for (int64_t i = dim_infos.size() - 1; i >= 0; --i) {
      // Default order goes: n-1, n-2, ..., 0
      dim_infos[i].order = i;
    }
    if (!permutation.empty()) {
      ICHECK_EQ(permutation.size(), dim_infos.size());
      std::vector<BufferDimIndexInfo> dim_infos_perm;
      for (size_t i = 0; i < permutation.size(); ++i) {
        dim_infos_perm.push_back(dim_infos[permutation[i]->value]);
      }
      dim_infos = std::move(dim_infos_perm);
    }
    return {offset, dim_infos};
  }

  /*!
   * \brief Match a K-dim region with N rows, pad it to MIN_DOT_ROWS rows, and use existing tools to
   * convert it to a block pointer.
   * \param region The region to pad.
   * \param check_n_dims The number of dimensions of the region. If not provided, any number of
   * dimensions is allowed.
   * \return (PrimExpr, PrimExpr)
   *    The first PrimExpr is the block pointer, which can be loaded from or stored to.
   *    The second PrimExpr is the mask for the block pointer.
   *    Returns std::nullopt if pattern matching fails.
   */
  std::optional<std::pair<PrimExpr, PrimExpr>> CreatePaddedRegionWithMask(
      const BufferRegion& region, std::optional<size_t> check_n_dims) {
    static const int ROW_DIM = 0;
    auto o2i_map = region->output_to_input_dims;
    if (global_buffers_.count(region->buffer->data) == 0) {
      return {};
    }
    size_t n_dims = o2i_map.size();
    if (check_n_dims.has_value() && n_dims != check_n_dims.value()) {
      return {};
    }
    // The row dim of matmul LHS cannot be an unsqueezed dim, it must map to a valid input dim.
    size_t row_input_dim = o2i_map[ROW_DIM].value()->value;
    Array<Range> ranges = region->region;
    auto row_range = ranges[row_input_dim];
    if (auto rows_int = as_const_int(row_range->extent); !rows_int || *rows_int >= MIN_DOT_ROWS) {
      return {};
    }
    // Create a new region that has MIN_DOT_ROWS rows, and use existing tools to convert it to a
    // block pointer.
    ranges.Set(row_input_dim, Range::FromMinExtent(row_range->min, MIN_DOT_ROWS));
    auto padded_block_ptr =
        MakeTritonBlockPtr(BufferRegion(region->buffer, ranges, o2i_map)).value();
    // Create a mask as tl.arange(0, MIN_DOT_ROWS) < rows
    auto row_arange = MakeAndLoadRangeBuffer(Integer(MIN_DOT_ROWS), n_dims, ROW_DIM);
    PrimExpr mask = row_arange < row_range->extent;
    return {{padded_block_ptr, mask}};
  }

  BufferRegion MakeAndLoadRangeBuffer(IntImm extent, size_t n_dims, size_t i_dim) {
    auto it = range_buffers_.find(extent->value);
    Buffer range_buffer;
    if (it == range_buffers_.end()) {
      auto dtype = extent->dtype;
      String new_buf_name =
          name_supply_->FreshName("prange", /*add_prefix=*/false, /*add_underscore=*/false);
      range_buffer = decl_buffer({extent}, dtype, new_buf_name, "shared");
      auto range_value = Call(dtype, tir::builtin::tile_arange(), {Integer(0), extent});
      range_buffers_[extent->value] =
          BufferRegionStore(BufferRegion::FullRegion(range_buffer), range_value);
    } else {
      range_buffer = it->second->region->buffer;
    }
    Array<Optional<Integer>> output_to_input_dims(n_dims, NullOpt);
    output_to_input_dims.Set(i_dim, Integer(0));
    return BufferRegion(range_buffer, {Range::FromMinExtent(0, extent)}, output_to_input_dims);
  }

  static const int MIN_DOT_ROWS = 16;

  const Map<Var, Buffer>& global_buffers_;
  NameSupply name_supply_;
  std::unordered_map<int64_t, BufferRegionStore> range_buffers_;
  // A mask that is created once we detect a dot operation to be padded.
  PrimExpr all_store_mask_;
};

class GridShapeCollector : public StmtVisitor {
 public:
  static std::tuple<Array<PrimExpr>, Integer, Integer> Visit(const Stmt& stmt) {
    GridShapeCollector col;
    col.VisitStmt(stmt);
    ICHECK(col.block_idx_x_.defined()) << "blockIdx.x is not defined";
    Array<PrimExpr> launch_args({col.block_idx_x_.value()});
    if (col.block_idx_y_.defined()) {
      launch_args.push_back(col.block_idx_y_.value());
    }
    if (col.block_idx_z_.defined()) {
      launch_args.push_back(col.block_idx_z_.value());
    }
    // NOTE: default values {num_warps=4, num_stages=3} here follow Triton's default values for CUDA
    // backend (triton/backends/nvidia/compiler.py).
    // See `compile_triton_source_to_device_module` in TVM Python.
    // We provide default values here so that users can have it without calling that function.
    return {launch_args, col.num_warps_.value_or(4), col.num_stages_.value_or(3)};
  }

 private:
  void VisitStmt_(const AttrStmtNode* attr_stmt) {
    if (attr_stmt->attr_key == attr::thread_extent) {
      auto iv = Downcast<IterVar>(attr_stmt->node);
      SetThreadBinding(iv, attr_stmt->value);
    } else if (attr_stmt->attr_key == attr::triton_num_warps) {
      CheckAndSet(num_warps_, "num_warps", Downcast<Integer>(attr_stmt->value));
    } else if (attr_stmt->attr_key == attr::triton_num_stages) {
      CheckAndSet(num_stages_, "num_stages", Downcast<Integer>(attr_stmt->value));
    }
    StmtVisitor::VisitStmt_(attr_stmt);
  }

  void SetThreadBinding(const IterVar& iter_var, PrimExpr extent) {
    if (iter_var->thread_tag == "blockIdx.x") {
      CheckAndSet(block_idx_x_, "blockIdx.x", extent);
    } else if (iter_var->thread_tag == "blockIdx.y") {
      CheckAndSet(block_idx_y_, "blockIdx.y", extent);
    } else if (iter_var->thread_tag == "blockIdx.z") {
      CheckAndSet(block_idx_z_, "blockIdx.z", extent);
    }
  }

  template <typename T>
  void CheckAndSet(Optional<T>& slot, const std::string& name, const T& expr) {
    if (slot.defined()) {
      ICHECK(ana_.CanProve(slot == expr))
          << "Grid size " << name << " is already set to a different value: " << slot << " vs. "
          << expr;
    } else {
      slot = expr;
    }
  }

  Optional<PrimExpr> block_idx_x_, block_idx_y_, block_idx_z_;
  Optional<Integer> num_warps_, num_stages_;
  arith::Analyzer ana_;
};

TritonSourceKernelInfo TVMFuncToTritonSource(const String& func_name,
                                             const DeviceFuncInfo& dev_func) {
  // Step 1. Apply TritonRealizePointer, to convert indexing operations on global buffers to
  // Triton's block pointer idiom.
  // NOTE: this pass also detects any dot operation that doesn't have enough rows (<16 rows) and pad
  // them. These are performed in the same pass, because this padding step needs to inform the
  // region-to-pointer decision.
  PrimFunc func = TritonRealizePointerMutator::VisitFunc(dev_func.func, dev_func.buffer_map);
  // Step 2. Apply GridShapeCollector, to collect the grid shape and the number of warps and stages
  // encoded in the function.
  auto [grid_shape, num_warps, num_stages] = GridShapeCollector::Visit(func->body);
  // Step 3. Apply FunctionToTritonScript to print the Triton kernel as a string of Python code.
  std::string source = codegen::FunctionToTritonScript(func_name, func);
  return TritonSourceKernelInfo{func_name, source, grid_shape, num_warps, num_stages};
}

}  // namespace

namespace transform {

Pass TritonBuildKernel() {
  return ::tvm::transform::CreateModulePass(
      [](IRModule m, PassContext ctx) { return TritonBuildKernelPass(std::move(m)); }, 0,
      "tir.TritonBuildKernel", {});
}

TVM_REGISTER_GLOBAL("tir.transform.TritonBuildKernel").set_body_typed(TritonBuildKernel);
TVM_REGISTER_GLOBAL("tir.transform.TritonCollectKernel").set_body_typed(TritonCollectKernel);

}  // namespace transform
}  // namespace tvm::tir
