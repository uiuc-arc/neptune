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

#include <tvm/arith/bound.h>
#include <tvm/support/iterator.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/utils.h>

#include "../analysis.h"
#include "../instruction_traits.h"
#include "../primitive.h"

namespace tvm {
namespace tir {

namespace {

BufferStore GetBlockBufferStore(const Block& block) {
  return block->body.as<BufferStore>().value_or(BufferStore());
}

Select MatchIfThenElse(const BufferStore& store) {
  auto* call = store->value.as<CallNode>();
  if (!call || !call->op.same_as(builtin::if_then_else())) {
    return Select();
  }
  PrimExpr condition = call->args[0];
  // We require that the false branch is a constant, and the condition is not trivially true.
  if (!is_const_number(call->args[2]) || is_const_int(condition, 1)) {
    return Select();
  }
  return Select(condition, call->args[1], call->args[2]);
}

Range SolveConditionAsLoopRange(const For& this_loop, const PrimExpr& cond,
                                const Map<Var, arith::IntSet>& loop_ranges,
                                arith::Analyzer& analyzer);

Stmt CopyLoopNestWithBlockBody(const std::vector<StmtSRef>& loop_srefs, BlockRealize br,
                               String new_name, Stmt new_body) {
  auto new_block = br->block;
  auto block_ptr = new_block.CopyOnWrite();
  block_ptr->name_hint = new_name;
  block_ptr->body = new_body;
  Stmt ret = BlockRealize(br->iter_values, br->predicate, std::move(new_block));
  for (auto it = loop_srefs.rbegin(); it != loop_srefs.rend(); ++it) {
    auto loop = GetRef<For>(TVM_SREF_TO_FOR(*it));
    loop.CopyOnWrite()->body = ret;
    ret = loop;
  }
  return ret;
}

//! \brief Allows replacing a Stmt with another Stmt.
struct StmtReplacer : public StmtExprMutator {
  static void Replace(ScheduleState self, const StmtSRef& sref_to_replace,
                      const Stmt& replacement) {
    auto stmt = GetRef<Stmt>(sref_to_replace->stmt);
    auto parent_sref = GetRef<StmtSRef>(sref_to_replace->parent);
    ICHECK(parent_sref.defined()) << "Stmt does not have a StmtSRef parent: " << stmt;
    auto new_parent_stmt = StmtReplacer(stmt, replacement)(GetRef<Stmt>(parent_sref->stmt));
    self->Replace(parent_sref, new_parent_stmt, {});
  }

 private:
  StmtReplacer(const Stmt& target_stmt, const Stmt& replacement)
      : target_stmt_(target_stmt), replacement_(replacement) {}

  Stmt VisitStmt(const Stmt& stmt) override {
    if (stmt.same_as(target_stmt_)) {
      return replacement_;
    }
    return StmtExprMutator::VisitStmt(stmt);
  }

  const Stmt& target_stmt_;
  Stmt replacement_;
};

}  // namespace

void PropagateIfThenElse(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                         String registered_handler) {
  arith::Analyzer analyzer;
  auto block_loops = GetLoops(block_sref);
  auto it = std::find(block_loops.begin(), block_loops.end(), loop_sref);
  ICHECK(it != block_loops.end())
      << "The given block is required to be under the given loop for PropagateIfThenElse to work";
  auto ExtractLoopIntSet = [](const StmtSRef& loop_sref) {
    auto loop = GetRef<For>(TVM_SREF_TO_FOR(loop_sref));
    return std::make_pair(loop->loop_var, arith::IntSet::FromMinExtent(loop->min, loop->extent));
  };
  std::vector<StmtSRef> inner_loops(it + 1, block_loops.end());
  auto inner_loop_ranges =
      support::map(inner_loops, ExtractLoopIntSet).to_container<Map<Var, arith::IntSet>>();
  auto all_loop_ranges =
      support::map(block_loops, ExtractLoopIntSet).to_container<Map<Var, arith::IntSet>>();

  BlockRealize br = GetBlockRealize(self, block_sref);
  Map<Var, PrimExpr> bindings = GetBindings(br);
  BufferStore store = GetBlockBufferStore(br->block);
  ICHECK(store.defined()) << "Block " << br->block->name_hint
                          << " is expected to have exactly one BufferStore";
  Select select = MatchIfThenElse(store);
  ICHECK(select.defined()) << "Block " << br->block->name_hint
                           << " does not match expected if-then-else pattern";
  PrimExpr cond = Substitute(select->condition, bindings);
  PrimExpr exist_cond = RelaxVarFromCondition(cond, inner_loop_ranges, RelaxationMode::kExists);
  auto loop = GetRef<For>(TVM_SREF_TO_FOR(loop_sref));
  Range new_loop_range = SolveConditionAsLoopRange(loop, exist_cond, all_loop_ranges, analyzer);
  if (!new_loop_range.defined()) {
    return;
  }
  auto* for_node = loop.CopyOnWrite();
  for_node->annotations.Set(attr::tir_loop_original_bounds,
                            Range::FromMinExtent(loop->min, loop->extent));
  for_node->min = new_loop_range->min;
  for_node->extent = new_loop_range->extent;
  self->Replace(loop_sref, loop, {});

  // Create an `if (T.likely(cond)) {cond_free_loop_nest} else {loop_nest}`, and replace the block
  // with it. `LoopPartition` will pick up this hint to create even more condition-free sections.
  // - Use kForAll to relax the `cond`, to get a predicate on when the block condition is _always_
  //   true (and therefore needs no masking operation).
  // TODO: if forall x in `new_loop_range`, `forall_cond` is constant false (or constant true, but
  // very unlikely), we should not create the if-then-else.
  PrimExpr forall_cond = RelaxVarFromCondition(cond, inner_loop_ranges, RelaxationMode::kForAll);
  // Create the new loop nest which wraps around the condition-free block.
  BufferStore uncond_store = store;
  uncond_store.CopyOnWrite()->value = select->true_value;
  auto new_loop_nest =
      CopyLoopNestWithBlockBody(inner_loops, br, br->block->name_hint + "_likely", uncond_store);
  // Create if-then-else, and replace the old loop nest with it.
  auto sref_to_replace = inner_loops.empty() ? block_sref : inner_loops.front();
  auto old_loop_nest = GetRef<Stmt>(sref_to_replace->stmt);
  StmtReplacer::Replace(self, sref_to_replace,
                        IfThenElse(likely(forall_cond), new_loop_nest, old_loop_nest));
}

namespace {

bool OnlyUsesVars(const PrimExpr& expr, const std::unordered_set<const VarNode*>& var_set) {
  bool only_uses_vars = true;
  PostOrderVisit(expr, [&](const ObjectRef& obj) {
    if (auto* var = obj.as<VarNode>()) {
      only_uses_vars = var_set.count(var);
    }
  });
  return only_uses_vars;
}

template <typename T, typename F>
void AsConjunctive(const PrimExpr& expr, std::vector<T>& conds, const F& f) {
  if (auto and_ = expr.as<AndNode>()) {
    AsConjunctive(and_->a, conds, f);
    AsConjunctive(and_->b, conds, f);
  } else {
    conds.push_back(f(expr));
  }
}

Range SolveConditionAsLoopRange(const For& this_loop, const PrimExpr& cond,
                                const Map<Var, arith::IntSet>& loop_ranges,
                                arith::Analyzer& analyzer) {
  std::unordered_set<const VarNode*> loop_vars;
  Map<Var, arith::IntSet> loop_doms;
  for (const auto& [var, int_set] : loop_ranges) {
    loop_vars.insert(var.get());
    loop_doms.Set(var, int_set);
    analyzer.Bind(var, int_set.CoverRange(Range()));
  }
  // The condition `cond` is a boolean expression `b(x, *v)`, where `x` is the current
  // loop's variable, and `*v` are outer loop vars. We now try to solve a bound on x: `lower(*v)
  // <= x < upper(*v)`.
  // Check if the condition contains the current loop var, and only uses available loop vars.
  Var lvar = this_loop->loop_var;
  if (!UsesVar(cond, [lvar](const VarNode* v) { return v == lvar.get(); }) ||
      !OnlyUsesVars(cond, loop_vars)) {
    return Range();
  }
  // Deduce-bound cannot work with multiple conditions, so we'll manually break down the
  // condition.
  std::vector<arith::IntSet> int_sets;
  AsConjunctive(cond, int_sets, [&](const PrimExpr& cond) {
    return arith::DeduceBound(lvar, cond, loop_doms, {});
  });
  // Add in the current loop's domain, and do an intersection.
  int_sets.push_back(arith::IntSet::FromMinExtent(this_loop->min, this_loop->extent));
  auto interval = arith::Intersect(int_sets);
  if (!interval.HasLowerBound() || !interval.HasUpperBound()) {
    return Range();
  }
  // Simplify the interval and return a range.
  PrimExpr new_min = analyzer.Simplify(interval.min());
  PrimExpr new_extent = analyzer.Simplify(interval.max() - new_min + 1);
  return Range::FromMinExtent(new_min, new_extent);
}

}  // namespace

struct PropagateIfThenElseTraits : public UnpackedInstTraits<PropagateIfThenElseTraits> {
  static constexpr const char* kName = "PropagateIfThenElse";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, LoopRV loop,
                                      String registered_handler) {
    sch->PropagateIfThenElse(block, loop, registered_handler);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, String loop,
                                 String registered_handler) {
    PythonAPICall py("propagate_if_then_else");
    py.Input("block", block);
    py.Input("loop", loop);
    py.Input("registered_handler", registered_handler);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(PropagateIfThenElseTraits);

}  // namespace tir
}  // namespace tvm