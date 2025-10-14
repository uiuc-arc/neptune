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

#include <tvm/arith/analyzer.h>
#include <tvm/ir/name_supply.h>
#include <tvm/runtime/thread_storage_scope.h>
#include <tvm/script/printer/doc.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include "../../relay/collage/utils.h"

namespace tvm::codegen {

namespace {
using namespace tir;
using namespace script::printer;

ExprDoc TritonPrefix(const String& attr) { return IdDoc("tl")->Attr(attr); }

ExprDoc KwargCallTriton(const String& name, const Array<ExprDoc>& args,
                        const std::vector<std::pair<String, ExprDoc>>& kwargs) {
  Array<String> kw_keys;
  Array<ExprDoc> kw_values;
  for (const auto& [key, value] : kwargs) {
    kw_keys.push_back(key);
    kw_values.push_back(value);
  }
  return TritonPrefix(name)->Call(args, kw_keys, kw_values);
}

ExprDoc CallTriton(const String& name, const Array<ExprDoc>& args) {
  return TritonPrefix(name)->Call(args);
}

ExprDoc CallTritonLibDevice(const String& name, const Array<ExprDoc>& args) {
  return IdDoc("libdevice")->Attr(name)->Call(args);
}

ExprDoc TritonDataType(const DataType& dtype) {
  ICHECK(dtype.is_scalar());
  if (dtype.is_int()) {
    return TritonPrefix("int" + std::to_string(dtype.bits()));
  } else if (dtype.is_float()) {
    return TritonPrefix("float" + std::to_string(dtype.bits()));
  } else if (dtype.is_e5m2_float8()) {
    return TritonPrefix("float8e5");
  } else if (dtype.is_e4m3_float8()) {
    return TritonPrefix("float8e4nv");
  }
  LOG(FATAL) << "Unsupported data type: " << dtype;
}

template <typename Iter>
ExprDoc IntoListDoc(const Iter& begin, const Iter& end) {
  return ListDoc({begin, end});
}

// Relevant information for a buffer in Triton codegen.
struct BufferInfo {
  // The scope of the buffer.
  ObjectRef scope;
  // The initialization of the buffer, if any.
  Optional<ExprDoc> init;
};

// Run type inference for what will become a Triton dot operation.
// TVM allows matmuls with arbitrary dtypes (T1 x T2 -> T3) by casting T1 and T2 to T3.
// Triton on the other hand requires the inputs to be the same dtype (T1 x T1 -> T2),
// and a list of allowed (T1, T2) combinations is largely defined by the hardware (tensorcore, etc.)
// Here we reconcile them two by casting the inputs to the appropriate dtype that Triton supports.
void MatmulDtypeCast(PrimExpr& lhs, PrimExpr& rhs, PrimExpr* acc, const DataType& out_dtype);

class CodeGenTriton : protected StmtFunctor<Doc(const Stmt&)>,
                      protected ExprFunctor<ExprDoc(const PrimExpr&)> {
 public:
  FunctionDoc VisitFunc(const std::string& func_name, const PrimFuncNode* func) {
    // 1. Add buffers in the function arguments to `buffer_params_`, and make a list of function
    // parameters in Python syntax.
    Array<AssignDoc> params;
    ICHECK(func->buffer_map.empty())
        << "Function `buffer_map` should be empty before using Triton codegen, and all buffers "
           "should have been lowered to using `decl_buffer`";
    for (auto param : func->params) {
      func_params_.insert(param.get());
      params.push_back(AssignDoc(VisitExpr_(param.get()), NullOpt, NullOpt));
    }
    // 2. Generate the function body.
    Array<StmtDoc> body = Flatten({VisitStmt(func->body)});
    // 3. Return the function.
    return FunctionDoc(IdDoc(func_name), params, {}, NullOpt, body);
  }

 protected:
  using ExprSelf = ExprFunctor<ExprDoc(const PrimExpr&)>;
  using StmtSelf = StmtFunctor<Doc(const Stmt&)>;

  using ExprSelf::VisitExpr;
  using StmtSelf::VisitStmt;

  Doc VisitStmt_(const BufferRegionStoreNode* store) override {
    // x[l:h] = y is probably not supported in Triton.
    ICHECK(store->region->IsFullRegion()) << "When storing to a buffer region, the buffer region "
                                             "must be a trivial full buffer region";
    ExprDoc region_doc = VisitExpr(store->region), value_doc = VisitExpr(store->value);
    // Get some buffer information, including scope info. If not found, the buffer is a function
    // parameter.
    auto it = buffer_infos_.find(store->region->buffer.get());
    if (it == buffer_infos_.end()) {
      return AssignDoc(region_doc, value_doc, NullOpt);
    }
    // Try to coalesce the first store to the buffer into the buffer initialization.
    // Conditions: value is constant, or only uses function parameters.
    auto buf_info = it->second;
    bool is_constant = true;
    PostOrderVisit(store->value, [this, &is_constant](const ObjectRef& expr) {
      if (auto var = expr.as<VarNode>(); var && !func_params_.count(var)) {
        is_constant = false;
      }
    });
    if (buf_info->scope.same_as(current_scope_) && !buf_info->init.defined() && is_constant) {
      buf_info->init = value_doc;
      // Skip code generation for this store.
      return StmtBlockDoc(Array<StmtDoc>());
    } else {
      return AssignDoc(region_doc, value_doc, NullOpt);
    }
  }

  Doc VisitStmt_(const EvaluateNode* op) override { return ExprStmtDoc(VisitExpr(op->value)); }

  ExprDoc VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::tile_full())) {
      // dtype, value, *shape
      String dtype_str = Downcast<StringImm>(op->args[0])->value;
      DataType dtype(runtime::String2DLDataType(dtype_str));
      double value = Downcast<FloatImm>(op->args[1])->value;
      Array<PrimExpr> shape{op->args.begin() + 2, op->args.end()};
      return CreateTritonFill(dtype, shape, value);
    } else if (op->op.same_as(builtin::triton_load())) {
      return CallTriton("load", VisitExprArray(op->args));
    } else if (op->op.same_as(builtin::triton_store())) {
      return CallTriton("store", VisitExprArray(op->args));
    } else if (op->op.same_as(builtin::tile_dot())) {
      auto args = std::vector<PrimExpr>(op->args.begin(), op->args.end());
      if (args.size() == 3) {
        MatmulDtypeCast(args[0], args[1], &args[2], op->dtype);
      } else if (args.size() == 2) {
        MatmulDtypeCast(args[0], args[1], nullptr, op->dtype);
      } else {
        LOG_FATAL << "tile_dot should have 2 or 3 arguments";
      }
      Array<ExprDoc> arg_docs = VisitExprArray(Array<PrimExpr>(args.begin(), args.end()));
      return CallTriton("dot", arg_docs);
    } else if (op->op.same_as(builtin::tile_reduce())) {
      ICHECK(op->args.size() == 3);
      auto reduce_fn = op->args[2].as<StringImmNode>();
      ICHECK(reduce_fn) << "reduce_fn must be a string";
      return CallTriton(reduce_fn->value, {VisitExpr(op->args[0]), VisitExpr(op->args[1])});
    } else if (op->op.same_as(builtin::triton_make_block_ptr())) {
      auto* n_dims_ = as_const_int(op->args[2]);
      ICHECK(n_dims_) << "n_dims must be a constant integer";
      size_t n_dims = *n_dims_;
      ICHECK_EQ(op->args.size(), n_dims * 5 + 3);

      auto arg_docs = VisitExprArray(op->args);
      OperationDoc ptr_offset_doc =
          OperationDoc(OperationDocNode::Kind::kAdd, {arg_docs[0], arg_docs[1]});
      auto base = arg_docs.begin() + 3;
      return KwargCallTriton("make_block_ptr", {ptr_offset_doc},
                             {{"shape", IntoListDoc(base, base + n_dims)},
                              {"strides", IntoListDoc(base + n_dims, base + n_dims * 2)},
                              {"offsets", IntoListDoc(base + n_dims * 2, base + n_dims * 3)},
                              {"block_shape", IntoListDoc(base + n_dims * 3, base + n_dims * 4)},
                              {"order", IntoListDoc(base + n_dims * 4, base + n_dims * 5)}});
    } else if (op->op.same_as(builtin::tile_arange())) {
      return CallTriton("arange", VisitExprArray(op->args));
    } else if (op->op.same_as(builtin::if_then_else())) {
      return CallTriton("where", VisitExprArray(op->args));
    } else if (op->op.same_as(builtin::tile_permute())) {
      Array<ExprDoc> dims;
      for (size_t i = 1; i < op->args.size(); ++i) {
        dims.push_back(VisitExpr(op->args[i]));
      }
      return CallTriton("permute", {VisitExpr(op->args[0]), ListDoc(dims)});
    } else if (op->op.same_as(builtin::triton_ptr_to_int())) {
      return VisitExpr(op->args[0]);
    }
    // Mathmatical functions:
    if (op->op.same_as(Op::Get("tir.exp"))) {
      return CallTriton("exp", {VisitExpr(op->args[0])});
    } else if (op->op.same_as(Op::Get("tir.rsqrt"))) {
      return CallTriton("rsqrt", {VisitExpr(op->args[0])});
    } else if (op->op.same_as(Op::Get("tir.tanh"))) {
      return CallTritonLibDevice("tanh", {VisitExpr(op->args[0])});
    }
    LOG_FATAL << "Unsupported call: " << op->op;
  }

  ExprDoc VisitExpr_(const BufferRegionNode* op) override {
    ExprDoc buffer = VisitExpr(op->buffer->data);
    auto unsqueeze_mask = op->AsUnsqueezeOnly();
    ICHECK(unsqueeze_mask) << "Only expand dim is supported in buffer region -- no actual indexing";
    bool no_unsqueeze = std::none_of(unsqueeze_mask->begin(), unsqueeze_mask->end(),
                                     [](bool is_expand_dim) { return is_expand_dim; });
    if (!no_unsqueeze) {
      Array<Doc> bounds;
      bounds.reserve(unsqueeze_mask->size());
      for (bool is_expand_dim : *unsqueeze_mask) {
        bounds.push_back(is_expand_dim ? Doc(LiteralDoc::None(NullOpt))
                                       : SliceDoc(NullOpt, NullOpt, NullOpt));
      }
      buffer = buffer[bounds];
    }
    return buffer;
  }

  ExprDoc VisitExpr_(const CastNode* op) override {
    return CallTriton("cast", {VisitExpr(op->value), TritonDataType(op->dtype)});
  }

  ExprDoc VisitExpr_(const MaxNode* op) override {
    return CallTriton("maximum", {VisitExpr(op->a), VisitExpr(op->b)});
  }
  ExprDoc VisitExpr_(const MinNode* op) override {
    return CallTriton("minimum", {VisitExpr(op->a), VisitExpr(op->b)});
  }

#define BINARY_OP_VISIT(Op, OpKind)                              \
  ExprDoc VisitExpr_(const Op##Node* op) override {              \
    ExprDoc a = VisitExpr(op->a);                                \
    ExprDoc b = VisitExpr(op->b);                                \
    return OperationDoc(OperationDocNode::Kind::OpKind, {a, b}); \
  }

  BINARY_OP_VISIT(Add, kAdd);
  BINARY_OP_VISIT(Sub, kSub);
  BINARY_OP_VISIT(Mul, kMult);
  BINARY_OP_VISIT(Div, kDiv);
  BINARY_OP_VISIT(FloorDiv, kFloorDiv);
  BINARY_OP_VISIT(FloorMod, kMod);
  BINARY_OP_VISIT(LT, kLt);
  BINARY_OP_VISIT(LE, kLtE);
  BINARY_OP_VISIT(EQ, kEq);
  BINARY_OP_VISIT(NE, kNotEq);
  BINARY_OP_VISIT(GT, kGt);
  BINARY_OP_VISIT(GE, kGtE);
  BINARY_OP_VISIT(And, kAnd);
  BINARY_OP_VISIT(Or, kOr);

  ExprDoc VisitExpr_(const IntImmNode* op) override { return LiteralDoc::Int(op->value, NullOpt); }
  ExprDoc VisitExpr_(const FloatImmNode* op) override {
    auto literal = LiteralDoc::Float(op->value, NullOpt);
    if (std::isinf(op->value) || std::isnan(op->value)) {
      return IdDoc("float")->Call({literal}, {}, {});
    }
    return literal;
  }
  ExprDoc VisitExpr_(const StringImmNode* op) override {
    return LiteralDoc::Str(op->value, NullOpt);
  }

  ExprDoc VisitExpr_(const VarNode* op) override {
    auto it = var_name_map_.find(GetRef<Var>(op));
    std::string name;
    if (it != var_name_map_.end()) {
      name = (*it).second;
    } else {
      name = name_supply_->FreshName(ProcessIdentifier(op->name_hint));
      var_name_map_.Set(GetRef<Var>(op), name);
    }
    return IdDoc(name);
  }

  /* Statements */

  Doc VisitStmt_(const ForNode* op) override {
    auto outer_scope = current_scope_;
    current_scope_ = GetRef<Stmt>(op);
    auto body = VisitStmt(op->body);
    current_scope_ = outer_scope;

    ICHECK(op->kind == ForKind::kSerial) << "Only serial loops are supported";
    auto max = analyzer_.Simplify(op->min + op->extent);
    auto range = IdDoc("range")->Call({VisitExpr(op->min), VisitExpr(max)});
    return ForDoc(VisitExpr(op->loop_var), range, Flatten({body}));
  }

  Doc VisitStmt_(const AttrStmtNode* op) override {
    auto body = VisitStmt(op->body);
    if (op->attr_key == tir::attr::thread_extent) {
      auto iv = Downcast<IterVar>(op->node);
      auto scope = runtime::ThreadScope::Create(iv->thread_tag);
      ICHECK(IsBlockIdx(scope)) << "Only block threads are supported";
      auto program_id = CallTriton("program_id", {LiteralDoc::Int(scope.dim_index, NullOpt)});
      auto pid_assign = AssignDoc(VisitExpr(iv->var), program_id, NullOpt);
      return StmtBlockDoc(Flatten({pid_assign, body}));
    }
    return body;
  }

  Doc VisitStmt_(const IfThenElseNode* op) override {
    auto outer_scope = current_scope_;
    current_scope_ = GetRef<Stmt>(op);
    Array<StmtDoc> then_case = Flatten({VisitStmt(op->then_case)});
    Array<StmtDoc> else_case;
    if (op->else_case) {
      else_case = Flatten({VisitStmt(op->else_case.value())});
    }
    current_scope_ = outer_scope;
    return IfDoc(VisitExpr(op->condition), then_case, else_case);
  }

  Doc VisitStmt_(const DeclBufferNode* op) override {
    Buffer buf = op->buffer;
    bool is_func_param = func_params_.count(buf->data.get());
    BufferInfo info{current_scope_, NullOpt};
    if (!is_func_param) {
      buffer_infos_.emplace(buf.get(), &info);
    }
    auto body = VisitStmt(op->body);
    if (is_func_param) {
      return body;
    }
    // The visitation of `op->body` may have filled `info.init`, which is the initialization value
    // of the buffer. See the BufferRegionStore visitor for when this happens.
    // If `info.init` is defined, add `buffer = init` before the body of this block.
    // It's guaranteed that either `info.init` is defined, or the buffer is assigned to at the
    // BufferRegionStore visitation. The latter is indistinguishable from a buffer definition,
    // because Python doesn't have variable declaration.
    buffer_infos_.erase(buf.get());
    ExprDoc buffer_name = VisitExpr(buf->data);
    if (info.init.defined()) {
      return StmtBlockDoc(Flatten({AssignDoc(buffer_name, info.init.value(), NullOpt), body}));
    } else {
      return body;
    }
  }

  Doc VisitStmt_(const AllocateNode* op) override { return VisitStmt(op->body); }

  Doc VisitStmt_(const SeqStmtNode* op) override {
    Array<Doc> stmts;
    for (auto stmt : op->seq) {
      stmts.push_back(VisitStmt(stmt));
    }
    return StmtBlockDoc(Flatten(stmts));
  }

  ExprDoc CreateTritonFill(const DataType& dtype, const Array<PrimExpr>& shape,
                           std::optional<float> value = std::nullopt) {
    ExprDoc shape_doc = ListDoc(VisitExprArray(shape));
    auto fill = CallTriton("zeros", {shape_doc, TritonDataType(dtype)});
    if (value.has_value() && value.value() != 0.0) {
      fill = OperationDoc(OperationDocNode::Kind::kAdd, {fill, VisitExpr(value.value())});
    }
    return fill;
  }

  Array<ExprDoc> VisitExprArray(const Array<PrimExpr>& op) {
    Array<ExprDoc> args;
    for (auto expr : op) {
      args.push_back(VisitExpr(expr));
    }
    return args;
  }

  Array<StmtDoc> Flatten(const Array<Doc>& docs) {
    Array<StmtDoc> stmts;
    for (auto doc : docs) {
      if (auto block = doc.as<StmtBlockDocNode>()) {
        stmts.insert(stmts.end(), block->stmts.begin(), block->stmts.end());
      } else {
        stmts.push_back(Downcast<StmtDoc>(doc));
      }
    }
    return stmts;
  }

  std::string ProcessIdentifier(std::string name) {
    std::replace(name.begin(), name.end(), '.', '_');
    auto idx = name.find("_shared");
    if (idx != std::string::npos) {
      name.replace(idx, 7, "");
    }
    idx = name.find("T_");
    if (idx != std::string::npos) {
      name.replace(idx, 2, "");
    }
    return name;
  }

  arith::Analyzer analyzer_;

  // The name supply for generating unique names for variables.
  // In TVM, variables in the same scope can have the same name, and TVM printer will distinguish
  // them, but this is not allowed in Python.
  NameSupply name_supply_;
  // A map from TVM variables to their names in the Triton script.
  Map<Var, String> var_name_map_;

  // Buffers that are passed as parameters to the function.
  std::unordered_set<const VarNode*> func_params_;
  // The current scope -- could be a for loop, an if-then-else, or nullptr (for the global scope).
  // This is used to track when a buffer allocation and an assignment happens.
  ObjectRef current_scope_;

  // Keep track of the buffer information for each buffer.
  // This only includes buffers that are NOT function parameters.
  std::unordered_map<const BufferNode*, BufferInfo*> buffer_infos_;
};

void MatmulDtypeCast(PrimExpr& lhs, PrimExpr& rhs, PrimExpr* acc, const DataType& out_dtype) {
  static const DataType float32 = DataType::Float(32), float16 = DataType::Float(16),
                        float8_1 = DataType::NVFloat8E4M3(), float8_2 = DataType::NVFloat8E5M2();
  auto CheckFloat32 = [out_dtype]() {
    ICHECK(out_dtype == float32) << "Triton matmul: float32 x float32 -> " << out_dtype
                                 << " unsupported";
  };
  // The matmul is a FP16 matmul if any of the inputs is FP16 (even if the other input is FP8),
  // and in this case we cast the other input to FP16.
  auto CheckAndCastFloat16 = [out_dtype](PrimExpr& expr) -> PrimExpr {
    ICHECK(out_dtype == float16 || out_dtype == float32)
        << "Triton matmul: float16 x float16 -> " << out_dtype << " unsupported";
    ICHECK(expr.dtype() == float16 || expr.dtype() == float32 || expr.dtype() == float8_1 ||
           expr.dtype() == float8_2)
        << "Input dtype " << expr.dtype() << " cannot be cast to float16 to perform float16 matmul";
    return cast(float16, expr);
  };

  auto lhs_dtype = lhs.dtype(), rhs_dtype = rhs.dtype();
  if (!lhs_dtype.is_scalar() || !rhs_dtype.is_scalar() || !out_dtype.is_scalar()) {
    LOG_FATAL << "Inputs and output must have scalars dtype in a Triton matmul";
  } else if (lhs_dtype == float32 && rhs_dtype == float32) {
    CheckFloat32();
  } else if (lhs_dtype == float16) {
    rhs = CheckAndCastFloat16(rhs);
  } else if (rhs_dtype == float16) {
    lhs = CheckAndCastFloat16(lhs);
  } else {
    // No implicit cast allowed for FP8 matmuls (same below).
    ICHECK((lhs_dtype == float8_1 && rhs_dtype == float8_1) ||
           (lhs_dtype == float8_2 && rhs_dtype == float8_2))
        << "Triton matmul: " << lhs_dtype << " x " << rhs_dtype << " -> " << out_dtype
        << " unsupported";
  }
  // If the accumulator is provided, cast it to the output dtype.
  if (acc) {
    *acc = cast(out_dtype, *acc);
  }
}

}  // namespace

String FunctionToTritonScript(const std::string& func_name, const PrimFunc& func) {
  return DocToPythonScript(CodeGenTriton().VisitFunc(func_name, func.get()), PrinterConfig());
}

}  // namespace tvm::codegen
