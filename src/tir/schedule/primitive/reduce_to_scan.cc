#include <tvm/tir/op.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/utils.h>

#include "../../../arith/pattern_match.h"
#include "../analysis.h"
#include "../error.h"
#include "../primitive.h"
#include "../transform.h"

namespace tvm::tir {

namespace {
class AxisOutOfRangeError : public ScheduleError {
 public:
  explicit AxisOutOfRangeError(IRModule mod, Buffer buffer, int factor_axis)
      : mod_(std::move(mod)), buffer_(std::move(buffer)), axis_(factor_axis) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `axis` is out of range. It is required to be in range "
           "[-(ndim + 1), ndim] where `ndim` is the number of dimensions of the write buffer";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    int ndim = static_cast<int>(buffer_->shape.size());
    os << "The write buffer " << buffer_->name << " has " << ndim
       << " dimension(s), so `axis` is required to be in [" << -(ndim + 1) << ", " << ndim
       << "] for rfactor. However, the input `axis` is " << axis_
       << ", which is out of the expected range";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  static int CheckAndUpdate(const IRModule& mod, const Buffer& buffer, int factor_axis) {
    int ndim = static_cast<int>(buffer->shape.size());
    if (factor_axis < -(ndim + 1) || factor_axis > ndim) {
      throw AxisOutOfRangeError(mod, buffer, factor_axis);
    }
    // If factor_axis is negative, convert it to a non-negative one.
    if (factor_axis < 0) {
      factor_axis += ndim + 1;
    }
    return factor_axis;
  }

  IRModule mod_;
  Buffer buffer_;
  int axis_;
};

// Replace:
// (1) all uses of `old_buf[...]` with `new_buf[..., size - 1, ...]`
// where `size` is the size of the new buffer at that dim; the new axis is inserted at `new_axis`.
// (2) the type tag on the iter vars, and the init stmt of the given block.
// (3) the given buffer store stmt, with another given buffer store stmt.
class ScanBufferReplacer : public StmtExprMutator {
 public:
  static std::pair<Block, Map<Block, Block>> RunOnBlock(const Block& root_block, const For& loop,
                                                        const Block& old_block,
                                                        const Block& new_block,
                                                        const Buffer& old_buf,
                                                        const Buffer& new_buf, int new_axis) {
    Map<Block, Block> block_sref_reuse;
    auto replacer = ScanBufferReplacer(loop, old_block, new_block, old_buf, new_buf, new_axis,
                                       block_sref_reuse);
    auto new_root = Downcast<Block>(replacer(root_block));
    return {new_root, block_sref_reuse};
  }

 private:
  explicit ScanBufferReplacer(const For& loop, const Block& old_block, const Block& new_block,
                              const Buffer& old_buf, const Buffer& new_buf, int new_axis,
                              Map<Block, Block>& block_sref_reuse)
      : loop_(loop),
        old_block_(old_block),
        new_block_(new_block),
        old_buf_(old_buf),
        new_buf_(new_buf),
        new_axis_(new_axis),
        block_sref_reuse_(block_sref_reuse) {}

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    auto op_ = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (op_->buffer.same_as(old_buf_)) {
      auto* op_ptr = op_.CopyOnWrite();
      op_ptr->buffer = new_buf_;
      op_ptr->indices = GetUpdatedIndices(op_ptr->indices);
      return GetRef<Stmt>(op_ptr);
    }
    return op_;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto op_ = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (op_->buffer.same_as(old_buf_)) {
      auto* op_ptr = op_.CopyOnWrite();
      op_ptr->buffer = new_buf_;
      op_ptr->indices = GetUpdatedIndices(op_ptr->indices);
      return GetRef<PrimExpr>(op_ptr);
    }
    return op_;
  }

  Stmt VisitStmt_(const BlockRealizeNode* br_node) override {
    // Block to block replacement takes priority.
    if (br_node->block.same_as(old_block_)) {
      block_sref_reuse_.Set(old_block_, new_block_);
      return BlockRealize(br_node->iter_values, br_node->predicate, new_block_);
    }

    // Create the new index expression we'll use for the new buffer at `new_buf_[new_axis_]`, which
    // may require modifying the block. We're doing it proactively, so we need a way to check
    // later if we actually don't need to do it (see below).
    auto br = GetRef<BlockRealize>(br_node);
    switch (ast_pos_) {
      case ASTPosition::kBeforeLoop:
        index_expr_ = loop_->min;
        break;
      case ASTPosition::kUnderLoop:
        std::tie(br, index_expr_) = SplitVarFromIterVars(
            br, loop_->loop_var, Range::FromMinExtent(loop_->min, loop_->extent));
        break;
      case ASTPosition::kAfterLoop:
        index_expr_ = loop_->min + loop_->extent - 1;
        break;
    }

    // Mutate various fields of the block.
    auto MutateRWRegion = [this](const BufferRegion& buffer_region) {
      if (buffer_region->buffer.same_as(old_buf_)) {
        auto region = buffer_region->region;
        region.insert(region.begin() + new_axis_, Range::FromMinExtent(index_expr_, 1));
        return BufferRegion(new_buf_, region);
      } else {
        return buffer_region;
      }
    };
    auto MutateMatchBuffer = [this](const MatchBufferRegion& match_buffer) {
      if (match_buffer->buffer.same_as(old_buf_)) {
        auto region = match_buffer->source->region;
        region.insert(region.begin() + new_axis_, Range::FromMinExtent(index_expr_, 1));
        return MatchBufferRegion(new_buf_, BufferRegion(new_buf_, region));
      } else {
        return match_buffer;
      }
    };
    auto block = br->block;
    Array<MatchBufferRegion> match_buffers = block->match_buffers.Map(MutateMatchBuffer);
    Array<BufferRegion> reads = block->reads.Map(MutateRWRegion);
    Array<BufferRegion> writes = block->writes.Map(MutateRWRegion);
    bool is_mutated = !reads.same_as(block->reads) || !writes.same_as(block->writes) ||
                      !match_buffers.same_as(block->match_buffers);
    // We detect if the old buffer appears in the block by checking if `match_buffers`, `reads`, or
    // `writes` are changed. We commit to block body and IterVar mutation only if so.
    if (!is_mutated) {
      return StmtMutator::VisitStmt_(br_node);
    }
    // Visit the block body.
    block = Downcast<Block>(StmtMutator::VisitStmt_(block.get()));
    {
      // Modify the block.
      BlockNode* n = block.CopyOnWrite();
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->match_buffers = std::move(match_buffers);
    }
    br.CopyOnWrite()->block = block;
    block_sref_reuse_.Set(br_node->block, block);
    return br;
  }

  Stmt VisitStmt_(const BlockNode* block_node) override {
    auto block = Downcast<Block>(StmtExprMutator::VisitStmt_(block_node));
    auto alloc_buffers = block_node->alloc_buffers;
    auto it = std::find(alloc_buffers.begin(), alloc_buffers.end(), old_buf_);
    if (it != alloc_buffers.end()) {
      alloc_buffers.erase(it);
      alloc_buffers.push_back(new_buf_);
      block.CopyOnWrite()->alloc_buffers = std::move(alloc_buffers);
      block_sref_reuse_.Set(GetRef<Block>(block_node), block);
    }
    return block;
  }

  Stmt VisitStmt_(const ForNode* for_node) override {
    bool is_target_loop = for_node->loop_var.same_as(loop_->loop_var);
    if (is_target_loop) {
      ast_pos_ = ASTPosition::kUnderLoop;
    }
    auto op_ = Downcast<For>(StmtExprMutator::VisitStmt_(for_node));
    if (is_target_loop) {
      ast_pos_ = ASTPosition::kAfterLoop;
    }
    return op_;
  }

  Array<PrimExpr> GetUpdatedIndices(Array<PrimExpr> indices) {
    indices.insert(indices.begin() + new_axis_, index_expr_);
    return indices;
  }

  enum class ASTPosition {
    kBeforeLoop,
    kUnderLoop,
    kAfterLoop,
  };

  const For& loop_;
  const Block &old_block_, &new_block_;
  const Buffer &old_buf_, &new_buf_;
  const int new_axis_;
  Map<Block, Block>& block_sref_reuse_;

  ASTPosition ast_pos_{ASTPosition::kBeforeLoop};
  PrimExpr index_expr_{};
};

class ScanBufferBlockCreator : public StmtExprMutator {
 public:
  static Block Create(const Block& block, const BufferStore& old_bufstore,
                      const BufferStore& new_bufstore, int axis) {
    ScanBufferBlockCreator replacer(old_bufstore, new_bufstore, axis);
    return Downcast<Block>(replacer(block));
  }

 private:
  ScanBufferBlockCreator(const BufferStore& old_bufstore, const BufferStore& new_bufstore, int axis)
      : old_bufstore_(old_bufstore), new_bufstore_(new_bufstore), axis_(axis) {}

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (op == old_bufstore_.get()) {
      buffer_var_map_.Set(new_bufstore_->buffer->data, new_bufstore_->buffer);
      return StmtExprMutator::VisitStmt_(new_bufstore_.get());
    }
    buffer_var_map_.Set(op->buffer->data, op->buffer);
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    buffer_var_map_.Set(op->buffer->data, op->buffer);
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BlockNode* block) override {
    Block mutated_block = Downcast<Block>(StmtMutator::VisitStmt_(block));
    BlockNode* n = mutated_block.CopyOnWrite();
    n->init = NullOpt;
    n->iter_vars.MutateByApply([](IterVar iv) {
      if (iv->iter_type == kCommReduce) {
        iv.CopyOnWrite()->iter_type = kOrdered;
      }
      return iv;
    });
    auto rw_regions = GetBlockReadWriteRegion(mutated_block, buffer_var_map_);
    n->reads = std::move(rw_regions[0]);
    n->writes = std::move(rw_regions[1]);
    return WithAnnotation(n, attr::tir_scan_buf_dim, Integer(axis_));
  }

  const BufferStore &old_bufstore_, new_bufstore_;
  int axis_;
  Map<Var, Buffer> buffer_var_map_;
};

// Check BufferStore(buffer, indices, expr) has the same usage for the LHS buffer -- all the
// BufferLoads in `expr` that load from `buffer` should have the same indices as `indices`.
// If so, create a substituter `f(expr) -> expr` that replaces each such load with a user-provided
// value.
std::function<PrimExpr(PrimExpr)> SubstBufferStoreLHS(const BufferStore& store) {
  Var placeholder_var("__buf_load", store->buffer->dtype);
  BufferLoad store_load(store->buffer, store->indices);
  auto subst_expr = ReplaceBufferLoads(store->value, [&](const BufferLoadNode* load) -> PrimExpr {
    auto load_ = GetRef<BufferLoad>(load);
    if (load_->buffer.same_as(store->buffer)) {
      ICHECK(arith::PConst<PrimExpr>(store_load).Match(load_));
      return placeholder_var;
    }
    return load_;
  });
  return
      [=](PrimExpr load) -> PrimExpr { return Substitute(subst_expr, {{placeholder_var, load}}); };
}

}  // namespace

void ReduceToScan(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                  int write_buffer_index, int axis) {
  // 1. Check that this block is a reduction block, then detect reduction pattern. Pattern-match the
  // BufferStore to confirm that it's a self reduction of form `output[*indices] =
  // f(output[*indices], ...)` for some `f`.
  StmtSRef scope_root = GetScopeRoot(self, block_sref,
                                     /*require_stage_pipeline=*/true);
  CheckReductionBlock(self, block_sref, scope_root);
  auto br = GetBlockRealize(self, block_sref);
  auto [init_values, updates] = GetInitValuesAndUpdatesFromReductionBlock(self, br->block);
  ICHECK(0 <= write_buffer_index && write_buffer_index < updates.size())
      << "Invalid write_buffer_index = " << write_buffer_index
      << "; expected 0 <= write_buffer_index < " << updates.size();
  PrimExpr init_value = init_values[write_buffer_index];
  BufferStore update = updates[write_buffer_index];
  axis = AxisOutOfRangeError::CheckAndUpdate(self->mod, update->buffer, axis);
  // Check that the BufferStore is a self reduction of form `output[*indices] = f(output[*indices],
  // ...)` for some `f`, then create a substituter `f(expr) -> expr` that replaces each
  // `output[*indices]` with another user-provided value.
  auto rhs_creator = SubstBufferStoreLHS(update);

  // 3. Get reduction IterVars from the block. Check that there is exactly one reduction IterVar
  // and that it maps to the loop var of the given `loop_sref`.
  // NOTE: we may be able to support multiple reduction IterVars as long as their expressions
  // combine to the loop var (for example, `vj0 = j // 32` and `vj1 = j % 32`).
  arith::Analyzer analyzer;
  std::vector<IterVar> reduce_ivs;
  {
    auto loop = TVM_SREF_TO_FOR(loop_sref);
    for (size_t i = 0; i < br->iter_values.size(); ++i) {
      auto& iv = br->block->iter_vars[i];
      if (iv->iter_type == kCommReduce) {
        ICHECK(analyzer.CanProveEqual(br->iter_values[i], loop->loop_var))
            << "The reduction IterVar " << iv->var << " does not map to the loop var "
            << loop->loop_var;
        reduce_ivs.push_back(iv);
      }
    }
  }
  ICHECK(reduce_ivs.size() == 1) << "There should be exactly one reduction IterVar in the block.";
  IterVar riv = reduce_ivs[0];
  // 4. Expand the output buffer by 1 dim with size that equals the extent of the IterVar. Also
  // compute the index expression to be used in the extra dim of the new buffer.
  PrimExpr extent = riv->dom->extent, index = riv->var;
  Buffer new_buffer = CreateExpandedBuffers({update}, axis, extent, ".scan")[0];
  // 5. Rebuild the BufferStore. We have matched its form to `output[*indices] =
  // f(output[*indices], rest)`, so we rebuild it as `output[..., index, ...] = if_then_else(index
  // > 0, f(output[..., index - 1, ...], init_value), rest)`.
  auto lhs_indices = update->indices, rhs_indices = update->indices;
  lhs_indices.insert(lhs_indices.begin() + axis, index);
  rhs_indices.insert(rhs_indices.begin() + axis, index - 1);
  PrimExpr new_comb_lhs = Select(index > 0, BufferLoad(new_buffer, rhs_indices), init_value);
  PrimExpr new_store_rhs = analyzer.Simplify(rhs_creator(new_comb_lhs));
  BufferStore new_store(new_buffer, new_store_rhs, lhs_indices);
  // 6. Replace the BufferStore in the block with our new one; update the type tags of the
  // `IterVar`s in the block from reduce to scan; remove the init stmt of the block;
  // add annotations "scan_buffer_axis": axis to the block.
  auto new_block = ScanBufferBlockCreator::Create(br->block, update, new_store, axis);
  // 7. Update all the uses of this buffer in the entire schedule. `ScanBufferReplacer` is also
  // responsible for replacing `block` with `new_block`.
  auto scope_block = GetRef<Block>(TVM_SREF_TO_BLOCK(scope_root));
  auto [new_scope_block, block_sref_reuse] =
      ScanBufferReplacer::RunOnBlock(scope_block, GetRef<For>(TVM_SREF_TO_FOR(loop_sref)),
                                     br->block, new_block, update->buffer, new_buffer, axis);
  self->Replace(scope_root, new_scope_block, block_sref_reuse);
  // FIXME: this isn't actually correct. The root block isn't necessarily stage-pipeline
  // because the scan block isn't necessarily region-cover.
  self->block_info.at(scope_root).stage_pipeline = true;
}

}  // namespace tvm::tir