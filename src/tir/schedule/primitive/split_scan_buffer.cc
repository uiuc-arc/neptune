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
#include "../../../arith/pattern_match.h"
#include "../instruction_traits.h"
#include "../utils.h"

namespace tvm {
namespace tir {

namespace {

class SplitScanBufferRewriter : public StmtExprMutator {
 public:
  static SplitScanBufferRewriter CreateRewriter(Block scan_block, Buffer scan_buffer, For scan_loop,
                                                size_t scan_dim) {
    std::string base_name = scan_buffer->name;
    if (auto pos = base_name.find(".scan"); pos != std::string::npos) {
      base_name = base_name.replace(pos, 5, "");
    }
    auto CopyBufferWithName = [&](std::string name) {
      Array<PrimExpr> new_shape = scan_buffer->shape;
      new_shape.erase(new_shape.begin() + scan_dim);
      auto new_buffer = make_object<BufferNode>(*scan_buffer.get());
      new_buffer->data = scan_buffer->data.copy_with_name(name);
      new_buffer->shape = std::move(new_shape);
      new_buffer->name = std::move(name);
      return Buffer(new_buffer);
    };
    Buffer new_buffer_prev = CopyBufferWithName(base_name + ".prev"),
           new_buffer_next = CopyBufferWithName(base_name + ".next");
    return SplitScanBufferRewriter(scan_block, scan_buffer, new_buffer_prev, new_buffer_next,
                                   scan_loop, scan_dim);
  }

 private:
  explicit SplitScanBufferRewriter(Block scan_block, Buffer scan_buffer, Buffer new_buffer_prev,
                                   Buffer new_buffer_next, For scan_loop, size_t scan_dim)
      : scan_buffer_(scan_buffer),
        new_buffer_prev_(new_buffer_prev),
        new_buffer_next_(new_buffer_next),
        scan_block_(scan_block),
        scan_loop_(scan_loop),
        scan_dim_(scan_dim) {}

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    auto outer_bindings = bindings_;
    bindings_ = GetBindings(GetRef<BlockRealize>(op));
    BlockRealize stmt = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    bindings_ = outer_bindings;
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    in_scan_block_ = block == scan_block_.get();
    Block stmt = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
    in_scan_block_ = false;
    if (!stmt.same_as(old_stmt)) {
      BlockNode* n = stmt.CopyOnWrite();
      auto inferred_regions = GetBlockReadWriteRegion(
          stmt,
          {{new_buffer_prev_->data, new_buffer_prev_}, {new_buffer_next_->data, new_buffer_next_}});
      n->reads = RewriteAccessRegion(n->reads, inferred_regions[0]);
      n->writes = RewriteAccessRegion(n->writes, inferred_regions[1]);
      block_reuse.Set(old_stmt, stmt);
    }
    {
      Array<Buffer> new_alloc_buffers;
      bool found_scan_buffer = false;
      for (const Buffer& buffer : stmt->alloc_buffers) {
        if (buffer == scan_buffer_) {
          new_alloc_buffers.push_back(new_buffer_prev_);
          new_alloc_buffers.push_back(new_buffer_next_);
          found_scan_buffer = true;
        } else {
          new_alloc_buffers.push_back(buffer);
        }
      }
      if (found_scan_buffer) {
        BlockNode* n = stmt.CopyOnWrite();
        n->alloc_buffers = std::move(new_alloc_buffers);
        block_reuse.Set(old_stmt, stmt);
      }
    }
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore stmt = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (stmt->buffer.same_as(scan_buffer_)) {
      BufferStoreNode* n = stmt.CopyOnWrite();
      RewriteBufferAccess(n->buffer, n->indices);
    }
    return std::move(stmt);
  }
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad stmt = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (stmt->buffer.same_as(scan_buffer_)) {
      BufferLoadNode* n = stmt.CopyOnWrite();
      RewriteBufferAccess(n->buffer, n->indices);
    }
    return std::move(stmt);
  }

  PrimExpr VisitExpr_(const SelectNode* op) final {
    auto select = Downcast<Select>(StmtExprMutator::VisitExpr_(op));
    // Detect match of `0 < j0`.
    auto loop_var = scan_loop_->loop_var;
    arith::PConst<PrimExpr> p0(IntImm(loop_var->dtype, 0));
    arith::PConst<PrimExpr> pvar(loop_var);
    if ((p0 < pvar).Match(Substitute(select->condition, bindings_))) {
      if (in_scan_block_) {
        ICHECK(!buf_init_const_.defined());
        buf_init_const_ = select->false_value;
      }
      // If `j0 - 1` is not used in BufferLoads inside this select, we can return the true value.
      bool has_v_minus_1 = false;
      PostOrderVisit(select->true_value, [&](const ObjectRef& obj) {
        if (auto expr = obj.as<PrimExpr>()) {
          has_v_minus_1 |= MatchScanVarMinus1(Substitute(expr.value(), bindings_));
        }
      });
      return has_v_minus_1 ? select : select->true_value;
    }
    return select;
  }

  void RewriteBufferAccess(Buffer& buffer, Array<PrimExpr>& indices) {
    auto index = Substitute(indices[scan_dim_], bindings_);
    // TODO: if `MatchScanDimMax` is true, we should actually also check if the buffer access is
    // outside of the scan loop.
    if (MatchScanVarMinus1(index) || MatchScanDimMax(index)) {
      buffer = new_buffer_prev_;
      indices.erase(indices.begin() + scan_dim_);
    } else if (MatchScanVar(index)) {
      buffer = new_buffer_next_;
      indices.erase(indices.begin() + scan_dim_);
    } else {
      LOG(WARNING) << buffer->name << "[" << indices << "]; index = " << index;
      failed_ = true;
    }
  }

  Array<BufferRegion> RewriteAccessRegion(const Array<BufferRegion>& old_access_regions,
                                          const Array<BufferRegion>& infered_access_regions) {
    Array<BufferRegion> new_access_regions;
    for (const BufferRegion& buffer_region : old_access_regions) {
      if (buffer_region->buffer.same_as(scan_buffer_)) {
        new_access_regions.insert(new_access_regions.end(), infered_access_regions.begin(),
                                  infered_access_regions.end());
      } else {
        new_access_regions.push_back(buffer_region);
      }
    }
    return new_access_regions;
  }

  bool MatchScanVarMinus1(const PrimExpr& expr) const {
    if (!expr->dtype.is_int()) {
      return false;
    }
    auto loop_var = scan_loop_->loop_var;
    arith::PConst<PrimExpr> p_var(loop_var), p_offset(IntImm(expr->dtype, 1));
    return (p_var - p_offset).Match(expr);
  }

  bool MatchScanVar(const PrimExpr& expr) const {
    // NOTE: if the loop is trivial (extent = 1), the loop var is 0.
    return (is_one(scan_loop_->extent) && is_const_int(expr, 0)) ||
           expr.same_as(scan_loop_->loop_var);
  }

  bool MatchScanDimMax(const PrimExpr& expr) {
    return analyzer_.CanProve(expr == scan_buffer_->shape[scan_dim_] - 1);
  }

 public:
  const Buffer scan_buffer_, new_buffer_prev_, new_buffer_next_;
  const Block scan_block_;
  const For scan_loop_;
  const size_t scan_dim_;

  arith::Analyzer analyzer_;
  Map<Block, Block> block_reuse;
  bool failed_{false};
  PrimExpr buf_init_const_;

 private:
  Map<Var, PrimExpr> bindings_;
  bool in_scan_block_{false};
};

template <typename Func>
std::pair<Stmt, Block> SurjectiveOnBuffer(const Buffer& buffer, Func body_func) {
  Array<IterVar> ivs;
  Array<Var> loop_vars;
  Array<PrimExpr> iv_binding, iv_vars;
  Array<Range> iv_var_ranges;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    auto size = buffer->shape[i];
    Var loop_var("ax" + std::to_string(i), size->dtype);
    loop_vars.push_back(loop_var);
    iv_binding.push_back(loop_var);
    Var iv_var("v_ax" + std::to_string(i), size->dtype);
    ivs.push_back(IterVar(Range(0, size), iv_var, IterVarType::kDataPar));
    iv_vars.push_back(iv_var);
    iv_var_ranges.push_back(Range::FromMinExtent(iv_var, 1));
  }
  auto block = body_func(ivs, iv_vars, iv_var_ranges);
  Stmt stmt = BlockRealize(iv_binding, const_true(), block);
  for (int i = ivs.size() - 1; i >= 0; --i) {
    stmt = For(loop_vars[i], 0, buffer->shape[i], ForKind::kSerial, stmt);
  }
  return {stmt, block};
}

Block InsertBeforeAfterBlock(Stmt before, Stmt after, Block root_block) {
  Array<Stmt> body_stmts;
  if (auto seq = root_block->body.as<SeqStmtNode>()) {
    body_stmts = seq->seq;
  } else {
    body_stmts = {std::move(root_block->body)};
  }
  auto* ptr = root_block.CopyOnWrite();
  body_stmts.insert(body_stmts.begin(), std::move(before));
  body_stmts.push_back(std::move(after));
  ptr->body = body_stmts.size() > 1 ? SeqStmt(std::move(body_stmts)) : std::move(body_stmts[0]);
  return root_block;
}

}  // namespace

Array<StmtSRef> SplitScanBuffer(ScheduleState self, const StmtSRef& block_sref,
                                const StmtSRef& loop_sref, int write_buffer_index) {
  auto block = GetRef<Block>(TVM_SREF_TO_BLOCK(block_sref));
  auto writes = block->writes;
  ICHECK(writes.size() == 1) << "The block must have exactly one write buffer";
  auto scan_dim = GetAnn<Integer>(block_sref, attr::tir_scan_buf_dim);
  ICHECK(scan_dim.defined()) << "The block does not have a scan buffer dimension annotation";
  auto scope_root_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  auto rewriter = SplitScanBufferRewriter::CreateRewriter(
      block, writes[0]->buffer, GetRef<For>(TVM_SREF_TO_FOR(loop_sref)), scan_dim.value()->value);
  Stmt scope_root = GetRef<Stmt>(scope_root_sref->stmt);
  Stmt new_scope_root = rewriter(scope_root);
  if (rewriter.failed_) {
    return {};
  }
  // Insert an initialization block for `new_buffer_prev_`.
  auto new_prev = rewriter.new_buffer_prev_;
  ICHECK(rewriter.buf_init_const_.defined());
  auto [init_stmt, init_block] = SurjectiveOnBuffer(
      new_prev, [&](Array<IterVar> iter_vars, Array<PrimExpr> indices, Array<Range> ranges) {
        BufferStore buf_store(new_prev, rewriter.buf_init_const_, indices);
        BufferRegion write_region(new_prev, ranges);
        return Block(std::move(iter_vars), {}, {write_region}, new_prev->name, buf_store);
      });
  // Insert a copy block from `new_buffer_next_` to `new_buffer_prev_`.
  auto new_next = rewriter.new_buffer_next_;
  auto [copy_stmt, copy_block] = SurjectiveOnBuffer(
      new_prev, [&](Array<IterVar> iter_vars, Array<PrimExpr> indices, Array<Range> ranges) {
        BufferStore buf_store(new_prev, BufferLoad(new_next, indices), indices);
        BufferRegion read_region(new_next, ranges), write_region(new_prev, ranges);
        return Block(std::move(iter_vars), {read_region}, {write_region}, new_next->name,
                     buf_store);
      });
  new_scope_root = InsertBeforeAfterBlock(init_stmt, copy_stmt, Downcast<Block>(new_scope_root));
  // Update schedule states.
  self->Replace(scope_root_sref, new_scope_root, rewriter.block_reuse);
  auto init_block_sref = self->stmt2ref.at(init_block.get()),
       copy_block_sref = self->stmt2ref.at(copy_block.get());
  // Use compute-at and reverse-compute-at to move the copy block before and after the loop.
  tir::ComputeAt(self, init_block_sref, GetRef<StmtSRef>(loop_sref->parent),
                 /*preserve_unit_loops=*/false);
  tir::ReverseComputeAt(self, copy_block_sref, loop_sref,
                        /*preserve_unit_loops=*/false);
  return {init_block_sref, copy_block_sref};
}

struct SplitScanBufferTraits : public UnpackedInstTraits<SplitScanBufferTraits> {
  static constexpr const char* kName = "SplitScanBuffer";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static Array<BlockRV> UnpackedApplyToSchedule(Schedule sch, BlockRV block, LoopRV loop,
                                                Integer write_buffer_index) {
    return sch->SplitScanBuffer(block, loop, write_buffer_index.IntValue());
  }

  static String UnpackedAsPython(Array<String> outputs, String block, String loop,
                                 Integer write_buffer_index) {
    PythonAPICall py("split_scan_buffer");
    py.Input("block", block);
    py.Input("loop", loop);
    py.Input("write_buffer_index", write_buffer_index);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(SplitScanBufferTraits);
}  // namespace tir
}  // namespace tvm
