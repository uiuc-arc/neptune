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

#include <tvm/arith/iter_affine_map.h>
#include <tvm/support/iterator.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/schedule/utils.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm::tir {

using namespace tvm::support;
using namespace tvm::arith;

namespace {

struct BlockMarker : public StmtExprMutator {
  BlockMarker(const std::unordered_set<const ForNode*>& loop_hints, BlockMap* block_map)
      : loop_hints_(loop_hints), block_map_(block_map) {}

 private:
  // Conditionally add a block around the body of the loop.
  // Never add a block around the loop itself.
  Stmt VisitStmt_(const ForNode* for_node) override {
    auto for_body = StmtExprMutator::VisitStmt(for_node->body);
    auto for_loop = GetRef<For>(for_node);
    auto WrapWithBlock = [this](Stmt body) {
      if (auto realize = body.as<BlockRealize>()) {
        return realize.value();
      }
      return this->WrapWithBlock(body);
    };

    // If the loop is explicitly set in `loop_hints`, we add a block around the body of the loop,
    // then put the loop around the block.
    if (loop_hints_.count(for_node)) {
      for_loop.CopyOnWrite()->body = WrapWithBlock(for_body);
      return for_loop;
    }
    // If the loop is a "thread binding" loop, we first check if its a blockIdx binding. Anything
    // else is not allowed.
    if (for_node->thread_binding.defined()) {
      auto scope = GetThreadScope(for_node);
      ICHECK(IsBlockIdx(scope)) << "Any non-blockIdx loop binding is unsupported in the lowering "
                                   "path to  codegen. Got "
                                << GetRef<For>(for_node);
      // Then if its inside is a "normal" loop, we add a block around the inner loop.
      if (auto inner_loop = for_node->body.as<ForNode>();
          inner_loop && !inner_loop->thread_binding.defined()) {
        for_loop.CopyOnWrite()->body = WrapWithBlock(for_body);
        return for_loop;
      }
    }
    for_loop.CopyOnWrite()->body = for_body;
    return for_loop;
  }

  Stmt VisitStmt_(const SeqStmtNode* seq_node) override {
    auto seq = StmtExprMutator::VisitStmt_(seq_node);
    return WrapWithBlock(seq);
  }

  Stmt VisitStmt_(const BlockNode* block_node) override {
    // If there is already a block where we'd like to add a block, don't do anything.
    // We can achieve this by checking the body of the block -- if it's a seqstmt, we just
    // skip our custom visitor and go to its body.
    // We don't skip the loop visitor, because it never adds block around the loop -- only the body
    // of the loop.
    auto block = GetRef<Block>(block_node);
    if (auto seq = block_node->body.as<SeqStmtNode>()) {
      block.CopyOnWrite()->body = StmtExprMutator::VisitStmt_(seq);
    } else {
      block = Downcast<Block>(StmtExprMutator::VisitStmt_(block_node));
    }
    if (block_map_) {
      block_map_->Insert(GetRef<Block>(block_node), block);
    }
    return block;
  }

  BlockRealize WrapWithBlock(Stmt body) {
    if (auto realize = body.as<BlockRealize>()) {
      return realize.value();
    }
    auto block = Block({}, {}, {}, "", body);
    added_blocks_.insert(block.get());
    return BlockRealize({}, const_true(), block);
  }

 public:
  const std::unordered_set<const ForNode*>& loop_hints_;
  BlockMap* block_map_;

  std::unordered_set<const BlockNode*> added_blocks_{};
};

class BufferMapTracker {
 protected:
  void EnterPrimFunc(PrimFunc prim_func) {
    for (auto& [_, buffer] : prim_func->buffer_map) {
      buffer_var_map_.Set(buffer->data, buffer);
    }
  }

  void EnterBlock(const BlockNode* block_node) {
    for (auto& buffer : block_node->alloc_buffers) {
      buffer_var_map_.Set(buffer->data, buffer);
    }
    for (auto& buffer : block_node->match_buffers) {
      buffer_var_map_.Set(buffer->buffer->data, buffer->buffer);
    }
  }

  void ExitBlock(const BlockNode* block_node) {
    for (auto& buffer : block_node->alloc_buffers) {
      buffer_var_map_.erase(buffer->data);
    }
    for (auto& buffer : block_node->match_buffers) {
      buffer_var_map_.erase(buffer->buffer->data);
    }
  }

  Map<Var, Buffer> buffer_var_map_{};
};

class AutoBlockizer : public StmtMutator, BufferMapTracker {
 public:
  AutoBlockizer(PrimFunc func, const std::unordered_set<const BlockNode*>& added_blocks,
                BlockMap* block_map)
      : added_blocks_(added_blocks), block_map_(block_map) {
    EnterPrimFunc(func);
  }

 private:
#define UPDATE_NODE(node, part_name, body) \
  if (!node->part_name.same_as(body)) {    \
    auto new_node = node.CopyOnWrite();    \
    new_node->part_name = body;            \
  }

  Stmt VisitStmt_(const BlockRealizeNode* br_node) override {
    auto block = br_node->block;
    if (!block->body->IsInstance<BufferStoreNode>()) {
      // If there are inner blocks, just recurse and return.
      ICHECK(br_node->iter_values.empty())
          << "Outer blocks cannot have iter vars (see block " << block->name_hint << ")";
      auto guard = block_with_loops_.emplace(br_node, VisitorStack<Var>{});
      return StmtMutator::VisitStmt_(br_node);
    }

    // This is an innermost block. We define the "inner loops" of this block to be the part between
    // this block and its outer block. An outer block should always exist if `BlockMarker` has
    // been run.
    auto block_with_loops = block_with_loops_.data();
    ICHECK(block_with_loops.size() > 0)
        << "Each inner block must be inside of some outer block, but block " << block->name_hint
        << " is not";
    auto [outer_block, inner_loop_vars] = block_with_loops.back();
    // The plan is to shrink the iter vars of the inner block (this block) to depend on only the
    // inner loop vars. The outer loop vars will be free-roaming in the index expressions of this
    // block. In a well-formed program, compact-buffer-allocation will remove most of these index
    // expressions.
    auto [marks, preds] =
        arith::SubspaceDivide(br_node->iter_values, loop_var_range_, inner_loop_vars.to_vector(),
                              br_node->predicate, analyzer_);
    ICHECK(is_const_int(preds.first, 1))
        << "Predicate of outer blocks must always be true. "
           "Instead, during subspace division, we got "
        << preds.first << " for outer block \"" << outer_block->block->name_hint
        << "\" and inner block \"" << block->name_hint << "\"";
    // We're safe to ignore `preds` from now on. Construct the new iter values and a substitution
    // for the old iter vars.
    ICHECK(marks.size() == br_node->iter_values.size());
    Array<PrimExpr> new_iter_values;
    Array<IterVar> new_iter_vars;
    Map<Var, PrimExpr> iv_substs;
    for (size_t i = 0; i < marks.size(); ++i) {
      auto [outer_mark, inner_mark] = marks[i];
      auto old_iv = block->iter_vars[i];
      PrimExpr inner_expr = NormalizeIterMapToExpr(inner_mark->source);
      PrimExpr outer_expr = NormalizeIterMapToExpr(outer_mark->source);
      auto inner_extent = inner_mark->extent;
      // If the *outer* mark is non-trivial, and this is not a spatial iter var, and the block has
      // init stmt, throw an error, because our blockization would change the behavior of the init
      // stmt.
      ICHECK(is_const_int(outer_mark->extent, 1) || old_iv->iter_type == IterVarType::kDataPar ||
             !block->init.defined())
          << "Blockizing a non-spatial iter var would change the behavior of the block's init "
             "stmt. Try using decompose_reduction in schedule to move away the init stmt. Block: "
          << GetRef<BlockRealize>(br_node);
      // Remove trivial iter vars.
      if (is_const_int(inner_extent, 1)) {
        ICHECK(is_const_int(inner_expr, 0))
            << "Trivial iter vars should be 0, got " << inner_mark->source;
        iv_substs.Set(old_iv->var, outer_expr);
      } else {
        // Update the extent of the iter var.
        old_iv.CopyOnWrite()->dom = Range::FromMinExtent(0, inner_extent);
        new_iter_vars.push_back(old_iv);
        new_iter_values.push_back(inner_expr);
        iv_substs.Set(old_iv->var, outer_expr * inner_extent + old_iv->var);
      }
    }
    // Run substitution on the block body, and update the iter vars.
    block = Downcast<Block>(Substitute(block, iv_substs));
    block.CopyOnWrite()->iter_vars = new_iter_vars;
    if (block_map_) {
      block_map_->Insert(br_node->block, block);
    }
    return BlockRealize(new_iter_values, preds.second, block);
  }

  Stmt VisitStmt_(const BlockNode* block_node) override {
    BufferMapTracker::EnterBlock(block_node);
    auto new_block = Downcast<Block>(StmtMutator::VisitStmt_(block_node));
    BufferMapTracker::ExitBlock(block_node);
    // Regenerate the read/write regions for blocks that were added by BlockMarker.
    if (added_blocks_.count(block_node)) {
      auto regions = GetBlockReadWriteRegion(new_block, buffer_var_map_);
      ICHECK(regions.size() == 2);
      auto block_ptr = new_block.CopyOnWrite();
      block_ptr->reads = regions[0];
      block_ptr->writes = regions[1];
    }
    if (block_map_) {
      block_map_->Insert(GetRef<Block>(block_node), new_block);
    }
    return new_block;
  }

  Stmt VisitStmt_(const ForNode* for_node) override {
    auto& [_, loop_vars] = block_with_loops_.top();
    auto guard = loop_vars.push(for_node->loop_var);
    auto ann_orig_bounds = GetAnn<Range>(for_node, attr::tir_loop_original_bounds);
    auto range = ann_orig_bounds.defined() ? ann_orig_bounds.value()
                                           : Range::FromMinExtent(for_node->min, for_node->extent);
    loop_var_range_.Set(for_node->loop_var, range);
    auto new_for = StmtMutator::VisitStmt_(for_node);
    loop_var_range_.erase(for_node->loop_var);
    return new_for;
  }

  const std::unordered_set<const BlockNode*>& added_blocks_;
  BlockMap* block_map_;

  using BlockWithLoops = std::pair<const BlockRealizeNode*, VisitorStack<Var>>;
  VisitorStack<BlockWithLoops> block_with_loops_{};

  Map<Var, Range> loop_var_range_{};
  Analyzer analyzer_{};
};

}  // namespace

PrimFunc TileExprAutoBlockize(PrimFunc func, Array<StmtSRef> loop_hints, BlockMap* block_map) {
  auto loop_hints_set = support::map(loop_hints, [](const StmtSRef& loop_hint) {
                          return TVM_SREF_TO_FOR(loop_hint);
                        }).to_container<std::unordered_set>();
  BlockMarker marker(loop_hints_set, block_map);
  auto body = marker(func->body);
  body = AutoBlockizer(func, marker.added_blocks_, block_map)(body);
  func.CopyOnWrite()->body = body;
  return func;
}

}  // namespace tvm::tir
