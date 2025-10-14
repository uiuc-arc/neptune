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
#include <tvm/arith/pattern.h>
#include <tvm/support/iterator.h>
#include <tvm/tir/sexpr_printer.h>

#include <algorithm>
#include <mutex>

#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief A helper class to create a new scope that contains decomposed init body
 * and replaced old reduction block.
 */
class DecomposeReductionBlockReplacer : public StmtMutator {
 public:
  /*!
   * \brief The open interface to users to call the helper class
   * \param old_scope_root The original block scope before decomposition
   * \param target_loop The loop we insert the decomposed init body before
   * \param decompose_body The decomposed init body
   * \param old_reduction_block The reduction block we want to decompose
   * \return The new block scope and the updated reduction block
   */
  static std::pair<Block, Block> Replace(Block old_scope_root, For target_loop,
                                         Stmt decomposed_body, Block old_reduction_block) {
    DecomposeReductionBlockReplacer replacer(std::move(target_loop), std::move(decomposed_body),
                                             std::move(old_reduction_block));
    return std::make_pair(Downcast<Block>(replacer(std::move(old_scope_root))),
                          replacer.new_reduction_block_);
  }

 private:
  explicit DecomposeReductionBlockReplacer(For target_loop, Stmt decomposed_body,
                                           Block old_reduction_block)
      : target_loop_(std::move(target_loop)),
        decomposed_body_(std::move(decomposed_body)),
        old_reduction_block_(std::move(old_reduction_block)) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt mutated_stmt = StmtMutator::VisitStmt_(loop);
    if (loop == target_loop_.get()) {
      return SeqStmt({decomposed_body_, mutated_stmt});
    } else {
      return mutated_stmt;
    }
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    if (block == old_reduction_block_.get()) {
      ObjectPtr<BlockNode> p_new_block = CopyOnWrite(block);
      p_new_block->name_hint = p_new_block->name_hint + "_update";
      p_new_block->init = NullOpt;
      // Add write regions back to read regions in update block.
      Array<BufferRegion> new_reads;
      std::unordered_set<const BufferNode*> read_bufs;
      for (const BufferRegion& read_access : block->reads) {
        read_bufs.insert(read_access->buffer.get());
      }
      for (const BufferRegion& write_access : block->writes) {
        if (read_bufs.find(write_access->buffer.get()) == read_bufs.end()) {
          new_reads.push_back(write_access);
        }
      }
      for (const BufferRegion& read_access : block->reads) {
        new_reads.push_back(read_access);
      }
      p_new_block->reads = new_reads;
      new_reduction_block_ = Block(p_new_block);
      return new_reduction_block_;
    } else {
      return StmtMutator::VisitStmt_(block);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* seq) final {
    Array<Stmt> new_stmts;
    new_stmts.reserve(seq->seq.size());
    for (const Stmt& old_stmt : seq->seq) {
      new_stmts.push_back(VisitStmt(old_stmt));
    }
    return SeqStmt::Flatten(new_stmts);
  }

 private:
  For target_loop_;
  Stmt decomposed_body_;
  Block old_reduction_block_;
  Block new_reduction_block_;
};

class LoopHeightError : public ScheduleError {
 public:
  static void CheckLoopHigherThanReduceLoops(const IRModule& mod, const BlockNode* block,
                                             const BlockRealizeNode* realize,
                                             const Array<StmtSRef>& loops,
                                             const StmtSRef& loop_sref) {
    for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
      // For each block var of type kCommReduce, check its binding
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
      if (iter_var->iter_type != IterVarType::kCommReduce) {
        continue;
      }
      for (const StmtSRef& higher_loop : loops) {
        // Only check loops not lower than the target loop
        if (higher_loop.same_as(loop_sref)) {
          break;
        }
        // loop_var of a higher loop shouldn't contain loop var
        const Var& loop_var = higher_loop->StmtAs<ForNode>()->loop_var;
        if (UsesVar(binding, [v = loop_var.get()](const VarNode* var) { return var == v; })) {
          const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
          throw LoopHeightError(mod, GetRef<For>(loop), GetRef<Block>(block));
        }
      }
    }
  }

  explicit LoopHeightError(IRModule mod, For loop, Block block)
      : mod_(std::move(mod)), loop_(std::move(loop)), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: decompose_reduction expect the loop to be higher than all the loops "
           "related to reduce block var";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: decompose_reduction expect the loop {0} to be higher than all the loops "
          "related to reduce block var of block {1}";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_, block_}; }

  IRModule mod_;
  For loop_;
  Block block_;
};

PrimExpr RemakePredicate(PrimExpr pred, const std::unordered_set<const VarNode*>& discarded_loops) {
  if (is_one(pred)) return Bool(true);
  PrimExpr new_pred = Bool(true);
  auto f = [&](const VarNode* var) { return discarded_loops.count(var); };
  arith::PVar<PrimExpr> lhs, rhs, rest;
  for (;;) {
    if ((rest && (lhs < rhs)).Match(pred)) {
      if (!UsesVar(lhs.Eval(), f)) new_pred = new_pred && (lhs.Eval() < rhs.Eval());
      pred = rest.Eval();
    } else if ((lhs < rhs).Match(pred)) {
      if (!UsesVar(lhs.Eval(), f)) new_pred = new_pred && (lhs.Eval() < rhs.Eval());
      break;
    } else {
      ICHECK(false) << "Unexpected predicate for reduction block";
    }
  }
  return new_pred;
}

StmtSRef DecomposeReduction(ScheduleState self, const StmtSRef& block_sref,
                            const StmtSRef& loop_sref) {
  /*!
   *  Check
   *    - block is a reduction block
   *    - loop is not lower than all the loops related to reduce block var
   *  Mutate
   *    - generate loops related to data par block vars
   *    - generate corresponding init block and update block
   */
  // Condition Checks and Information Collection
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  // Get the outer loops from high to low
  Array<StmtSRef> loops = GetLoops(block_sref);
  const BlockRealizeNode* realize = GetBlockRealize(self, block_sref).get();
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref,
                                          /*require_stage_pipeline=*/false);
  if (self->enable_check) {
    // Cond 0. Check loop_sref is an ancestor of block_sref
    if (std::find(loops.begin(), loops.end(), loop_sref) == loops.end()) {
      throw LoopPositionError(self->mod, GetRef<For>(loop), GetRef<Block>(block),
                              "decompose_reduction");
    }
    // Cond 1. Check block is reduction
    CheckReductionBlock(self, block_sref, scope_root_sref);
    // Cond 2. Check 'loop' is higher than all the loops related to block var of type reduction
    LoopHeightError::CheckLoopHigherThanReduceLoops(self->mod, block, realize, loops, loop_sref);
  }
  // IR Manipulation
  ObjectPtr<BlockNode> init_block = make_object<BlockNode>();
  ObjectPtr<BlockRealizeNode> init_realize = make_object<BlockRealizeNode>();
  init_block->name_hint = block->name_hint + "_init";
  init_block->annotations = block->annotations;
  init_realize->iter_values = {};
  init_realize->block = Block(init_block);
  // Step 1. Create new block vars and their bindings
  // Maps an old block var to the new corresponding block var
  std::unordered_map<Var, Var> block_var_map;
  block_var_map.reserve(block->iter_vars.size());
  for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->iter_values[i];
    // Only process data parallel block vars
    if (iter_var->iter_type != IterVarType::kDataPar) {
      continue;
    }
    // Create a new block var
    IterVar new_iter_var(/*dom=*/iter_var->dom,
                         /*var=*/iter_var->var.copy_with_suffix(""),
                         /*iter_type=*/iter_var->iter_type,
                         /*thread_tag=*/iter_var->thread_tag);
    // Add a block var and its binding
    init_block->iter_vars.push_back(new_iter_var);
    init_realize->iter_values.push_back(binding);
    // Add a mapping from old block vars to new block vars
    block_var_map[iter_var->var] = new_iter_var->var;
  }
  // Step 2. After copying block vars, substitute them in init block
  init_block->body = Substitute(block->init.value(), block_var_map);
  for (const BufferRegion& write : block->writes) {
    init_block->writes.push_back(
        BufferRegion(write->buffer, Substitute(write->region, block_var_map)));
  }
  // Step 3. Scan loops not higher than the specified loop above the reduction block.
  //         If the loop is used in the init block binding, then it is chosen.
  //         Otherwise, it is discarded.
  std::unordered_set<const VarNode*> discarded_loops;
  std::vector<int> chosen_loops;
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
    const VarNode* loop_var = loops[i]->StmtAs<ForNode>()->loop_var.get();
    bool discarded = true;
    for (const PrimExpr& expr : init_realize->iter_values) {
      if (!UsesVar(expr, [v = loop_var](const VarNode* var) { return var == v; })) {
        continue;
      }
      // The loop is related to init block bindings;
      chosen_loops.push_back(i);
      discarded = false;
      break;
    }
    if (discarded) discarded_loops.insert(loop_var);
    // Only scan loops not higher than the given loop
    if (loops[i].same_as(loop_sref)) {
      break;
    }
  }
  // Step 4. After scanning loops, make a new predicate in the init block realize
  //         We discard predicate that is related to discarded loops
  init_realize->predicate = RemakePredicate(realize->predicate, discarded_loops);
  // Step 5. Create new loops above init block
  std::unordered_map<Var, Var> loop_var_map;
  Stmt body = BlockRealize(init_realize);
  for (int i : chosen_loops) {
    const ForNode* old_loop = TVM_SREF_TO_FOR(loops[i]);
    // Create a new equivalent to the chosen loop
    Var old_loop_var = old_loop->loop_var;
    Var new_loop_var = old_loop_var.copy_with_suffix("_init");
    loop_var_map[old_loop_var] = new_loop_var;
    Optional<IterVar> opt_thread_binding = old_loop->thread_binding;
    if (opt_thread_binding) {
      auto thread_binding = opt_thread_binding.value();
      auto new_var = thread_binding->var.copy_with_suffix("");
      thread_binding.CopyOnWrite()->var = new_var;
      opt_thread_binding = thread_binding;
    }
    body = For(/*loop_var=*/new_loop_var,
               /*min=*/old_loop->min,
               /*extent=*/old_loop->extent,
               /*kind=*/old_loop->kind,
               /*body=*/body,
               /*thread_binding=*/opt_thread_binding);
  }
  body = Substitute(body, loop_var_map);
  // Step 6. Mutate IR
  const BlockNode* old_scope_root = TVM_SREF_TO_BLOCK(scope_root_sref);
  auto [new_scope_root, new_reduction_block] = DecomposeReductionBlockReplacer::Replace(
      GetRef<Block>(old_scope_root), GetRef<For>(loop), body, GetRef<Block>(block));
  self->Replace(scope_root_sref, new_scope_root,
                {{GetRef<Block>(old_scope_root), new_scope_root},
                 {GetRef<Block>(block), new_reduction_block}});
  self->UpdateScopeBlockInfo(new_scope_root);
  return self->stmt2ref.at(init_block.get());
}

/******** Commutative Reducer ********/

/*!
 * \brief A structure used for registering new commutative reducers, and store all the registered
 * reducers. The reducers are preserved in a list, in the form of "reducer-getter function". When
 * invoking a reducer-getter function with a specific datatype, the reducer-getter will return the
 * CommReducer of the corresponding reduction pattern and the specific datatype
 */
struct ReducerRegistry {
  ReducerRegistry()
      : reducer_getters{
            CreateReducerGetter(
                /*n_buffers=*/1,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{x[0] + y[0]};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, 0)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/1,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{x[0] * y[0]};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, 1)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/1,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{min(x[0], y[0])};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{max_value(values[0]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/1,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{max(x[0], y[0])};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{min_value(values[0]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{x[0] + y[0], x[1] + y[1]};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, 0),
                                         make_const(values[1]->dtype, 0)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  PrimExpr idx = Select(x[1] >= y[1], x[0], y[0]);
                  PrimExpr val = Select(x[1] >= y[1], x[1], y[1]);
                  return Array<PrimExpr>{idx, val};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, -1),
                                         min_value(values[1]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  PrimExpr idx =
                      Select(Or(greater(x[1], y[1]), And(equal(x[1], y[1]), less(x[0], y[0]))),
                             x[0], y[0]);
                  PrimExpr val = Select(greater(x[1], y[1]), x[1], y[1]);
                  return Array<PrimExpr>{idx, val};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, -1),
                                         min_value(values[1]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  PrimExpr idx = Select(x[1] <= y[1], x[0], y[0]);
                  PrimExpr val = Select(x[1] <= y[1], x[1], y[1]);
                  return Array<PrimExpr>{idx, val};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, -1),
                                         max_value(values[1]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  PrimExpr idx = Select(
                      Or(less(x[1], y[1]), And(equal(x[1], y[1]), less(x[0], y[0]))), x[0], y[0]);
                  PrimExpr val = Select(less(x[1], y[1]), x[1], y[1]);
                  return Array<PrimExpr>{idx, val};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, -1),
                                         max_value(values[1]->dtype)};
                })} {}

  static void RegisterReducer(
      int n_buffers, TypedPackedFunc<Array<PrimExpr>(Array<Var>, Array<Var>)> combiner_getter,
      TypedPackedFunc<Array<PrimExpr>(Array<PrimExpr>)> identity_getter) {
    ReducerRegistry::Global()->reducer_getters.push_back(ReducerRegistry::CreateReducerGetter(
        n_buffers, std::move(combiner_getter), std::move(identity_getter)));
  }

  static TypedPackedFunc<Optional<CommReducer>(Array<PrimExpr>)> CreateReducerGetter(
      int n_buffers, TypedPackedFunc<Array<PrimExpr>(Array<Var>, Array<Var>)> combiner_getter,
      TypedPackedFunc<Array<PrimExpr>(Array<PrimExpr>)> identity_getter) {
    return [n_buffers,                                     //
            combiner_getter = std::move(combiner_getter),  //
            identity_getter = std::move(identity_getter)   //
    ](Array<PrimExpr> values) -> Optional<CommReducer> {
      if (static_cast<int>(values.size()) != n_buffers) {
        return NullOpt;
      }
      Array<Var> lhs;
      Array<Var> rhs;
      for (int i = 0; i < n_buffers; ++i) {
        lhs.push_back(Var("x" + std::to_string(i), values[i]->dtype));
        rhs.push_back(Var("y" + std::to_string(i), values[i]->dtype));
      }
      return CommReducer(lhs, rhs, combiner_getter(lhs, rhs), identity_getter(values));
    };
  }

  static ReducerRegistry* Global() {
    static ReducerRegistry instance;
    return &instance;
  }

  std::vector<TypedPackedFunc<Optional<CommReducer>(Array<PrimExpr>)>> reducer_getters;
};

std::vector<TypedPackedFunc<Optional<CommReducer>(Array<PrimExpr>)>> GetReducerGetters() {
  return ReducerRegistry::Global()->reducer_getters;
}

class NotSerialLoopKindError : public ScheduleError {
 public:
  explicit NotSerialLoopKindError(IRModule mod, For loop)
      : mod_(std::move(mod)), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The input loop of rfactor is required to be `kSerial`";
  }

  String DetailRenderTemplate() const final {
    String str_kind = ForKind2String(loop_->kind);
    std::ostringstream os;
    os << "ScheduleError: The input loop {0} of rfactor is required to be `Serial`. However, the "
          "kind of {0} is `"
       << str_kind << "`";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class FactorAxisOutOfRangeError : public ScheduleError {
 public:
  explicit FactorAxisOutOfRangeError(IRModule mod, Buffer buffer, int factor_axis)
      : mod_(std::move(mod)), buffer_(std::move(buffer)), factor_axis_(factor_axis) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `factor_axis` is out of range. It is required to be in range "
           "[-(ndim + 1), ndim] where `ndim` is the number of dimensions of the write buffer";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    int ndim = static_cast<int>(buffer_->shape.size());
    os << "The write buffer " << buffer_->name << " has " << ndim
       << " dimension(s), so `factor_axis` is required to be in [" << -(ndim + 1) << ", " << ndim
       << "] for rfactor. However, the input `factor_axis` is " << factor_axis_
       << ", which is out of the expected range";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  static int CheckAndUpdate(const IRModule& mod, const Buffer& buffer, int factor_axis) {
    int ndim = static_cast<int>(buffer->shape.size());
    if (factor_axis < -(ndim + 1) || factor_axis > ndim) {
      throw FactorAxisOutOfRangeError(mod, buffer, factor_axis);
    }
    // If factor_axis is negative, convert it to a non-negative one.
    if (factor_axis < 0) {
      factor_axis += ndim + 1;
    }
    return factor_axis;
  }

  IRModule mod_;
  Buffer buffer_;
  int factor_axis_;
};

class LoopPropertyError : public ScheduleError {
 public:
  enum ErrorType {
    kDataParIterTouchRFactorLoop = 0,
    kLoopTouchedByBothKindsOfBlockIters = 1,
    kNotFirstChildBlockOfOutermostLoop = 2,
    kUnboundLoopUnderReductionLoop = 3
  };

  explicit LoopPropertyError(IRModule mod, For loop, ErrorType error_type)
      : mod_(std::move(mod)), loop_(std::move(loop)), error_type_(error_type) {}

  String FastErrorString() const final {
    switch (error_type_) {
      case kDataParIterTouchRFactorLoop:
        return "ScheduleError: The loop to be applied rfactor is required not to be touched by any "
               "data parallel block iter of the block";
      case kLoopTouchedByBothKindsOfBlockIters:
        return "ScheduleError: The loops outside of the reduction block are required not to be "
               "touched by both data parallel block iters and reduction block iters";
      case kNotFirstChildBlockOfOutermostLoop:
        return "ScheduleError: The reduction block should be the first child block of the "
               "outermost loop outside of it";
      case kUnboundLoopUnderReductionLoop:
        return "ScheduleError: A loop who has extent greater than one and is not bound to any "
               "block iter should not appear under a reduction loop";
    }
    ICHECK(false) << "Unreachable";
    throw;
  }

  String DetailRenderTemplate() const final {
    switch (error_type_) {
      case kDataParIterTouchRFactorLoop:
        return "The loop to be applied rfactor is {0}, which is required not to be touched by any "
               "data parallel block iter of the block below. However, some of the block's data "
               "parallel block iters touch this loop";
      case kLoopTouchedByBothKindsOfBlockIters:
        return "It is not allowed that the loop {0} is touched by both some data parallel block "
               "iters and some reduction block iters";
      case kNotFirstChildBlockOfOutermostLoop:
        return "The first child block of the outermost loop {0} is not the reduction block.";
      case kUnboundLoopUnderReductionLoop:
        return "The loop {0} has extent greater than one, and is not bound to any block iter. "
               "Therefore it shouldn't appear under a reduction loop";
    }
    ICHECK(false) << "Unreachable";
    throw;
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  static void CheckLoopProperty(const ScheduleState& self, const Array<For>& loops,
                                const ForNode* rf_loop, const Block& block,
                                const std::unordered_set<const VarNode*>& data_par_loop_vars,
                                const std::unordered_set<const VarNode*>& reduce_loop_vars) {
    Array<BlockRealize> children_of_outermost_loop =
        GetChildBlockRealizeOnSRefTree(self->stmt2ref.at(loops[0].get()));
    if (!children_of_outermost_loop[0]->block.same_as(block)) {
      throw LoopPropertyError(self->mod, loops[0], kNotFirstChildBlockOfOutermostLoop);
    }

    bool meet_reduction_loop = false;
    for (const For& loop : loops) {
      bool data_par_touched = data_par_loop_vars.count(loop->loop_var.get());
      bool reduction_touched = reduce_loop_vars.count(loop->loop_var.get());

      if (data_par_touched && reduction_touched) {
        throw LoopPropertyError(self->mod, loop, kLoopTouchedByBothKindsOfBlockIters);
      } else if (data_par_touched) {
        if (loop.get() == rf_loop) {
          throw LoopPropertyError(self->mod, loop, kDataParIterTouchRFactorLoop);
        }
        continue;
      } else if (reduction_touched) {
        if (!meet_reduction_loop) {
          CheckGetSingleChildBlockRealizeOnSRefTree(self, self->stmt2ref.at(loop.get()));
          meet_reduction_loop = true;
        }
        continue;
      } else if (meet_reduction_loop && !is_one(loop->extent)) {
        throw LoopPropertyError(self->mod, loop, kUnboundLoopUnderReductionLoop);
      }
    }
  }

  IRModule mod_;
  For loop_;
  ErrorType error_type_;
};

/*!
 * \brief For each loop in the given array of loop, associate its loop var with the loop itself
 * using a mapping
 * \param loops The loops to be analyzed
 * \return A mapping from loops to their corresponding loop vars
 */
std::unordered_map<const VarNode*, For> GetLoopVar2LoopMap(const Array<For>& loops) {
  std::unordered_map<const VarNode*, For> loop_vars2loop;
  loop_vars2loop.reserve(loops.size());
  for (const For& loop : loops) {
    loop_vars2loop[loop->loop_var.get()] = loop;
  }
  return loop_vars2loop;
}

/*!
 * \brief The base class of the rfactor/write-back block creator, which creates the blocks in four
 * steps:
 * 1) Create the new block iters and the their iter bindings
 * 2) Create the body and init of the new block
 * 3) Create the read/write regions of the new block
 * 4) Create the new block and the new block-realize
 */
class BaseBlockCreator {
 public:
  explicit BaseBlockCreator(BlockRealize old_block_realize, For rf_loop,
                            Array<BufferStore> old_reduction_updates, CommReducer reducer,
                            Array<Buffer> rf_buffers, bool is_rf_block)
      : old_block_realize_(std::move(old_block_realize)),
        rf_loop_(std::move(rf_loop)),
        old_reduction_updates_(std::move(old_reduction_updates)),
        reducer_(std::move(reducer)),
        rf_buffers_(std::move(rf_buffers)),
        n_buffers_(static_cast<int>(rf_buffers_.size())),
        is_rf_block_(is_rf_block) {
    n_block_iters_ = static_cast<int>(old_block_realize_->iter_values.size());
    update_buffers_.reserve(n_buffers_);
    update_indices_.reserve(n_buffers_);
    update_lhs_.reserve(n_buffers_);
    update_rhs_.reserve(n_buffers_);
  }

  BlockRealize CreateBlock() {
    CreateAdditionalIter();
    for (int i = 0; i < n_block_iters_; ++i) {
      CreateNormalIters(i);
    }
    bool has_reduce_iter = false;
    for (const IterVar& iter_var : iter_vars_) {
      if (iter_var->iter_type == IterVarType::kCommReduce) {
        has_reduce_iter = true;
        break;
      }
    }

    // The pre-processing finds out the buffers written in the block, the indices of the buffer
    // accesses, and the reduction LHS and RHS of the stored values.
    PreProcess();
    Stmt block_body = Substitute(CreateBlockBody(has_reduce_iter), var_map_);
    Optional<Stmt> block_init = CreateBlockInit(has_reduce_iter);
    if (block_init.defined()) {
      block_init = Substitute(block_init.value(), var_map_);
    }
    CreateReadWriteRegions();

    String new_block_name = old_block_realize_->block->name_hint;
    PrimExpr predicate = const_true();
    if (is_rf_block_) {
      new_block_name = new_block_name + "_rf";
      predicate = old_block_realize_->predicate;
    }
    Block new_block(
        /*iter_vars=*/iter_vars_,
        /*reads=*/read_regions_,
        /*writes=*/write_regions_,
        /*name_hint=*/new_block_name,
        /*body=*/std::move(block_body),
        /*init=*/std::move(block_init),
        /*alloc_buffers=*/{},
        /*match_buffers=*/{},
        /*annotations=*/old_block_realize_->block->annotations);
    return BlockRealize(iter_values_, predicate, new_block);
  }

  Map<Var, PrimExpr> GetVarMap() const { return var_map_; }

 private:
  virtual void CreateAdditionalIter() = 0;
  virtual void CreateNormalIters(int idx) = 0;
  virtual void PreProcess() = 0;
  virtual void CreateReadWriteRegions() = 0;

  Stmt CreateBlockBody(bool has_reduce_iter) {
    Array<Stmt> buf_stores;
    buf_stores.reserve(n_buffers_);

    // Case 1. If the block has no reduction iterator, we just store the RHS values into the
    // buffers.
    if (!has_reduce_iter) {
      for (int i = 0; i < n_buffers_; ++i) {
        buf_stores.push_back(BufferStore(update_buffers_[i], update_rhs_[i], update_indices_[i]));
      }
      return n_buffers_ > 1 ? SeqStmt(buf_stores) : buf_stores[0];
    }

    // Case 2. If the reduction is for single buffer, the block body is a single BufferStore.
    Array<PrimExpr> stored_values = (*reducer_.get())(update_lhs_, update_rhs_);
    if (n_buffers_ == 1) {
      return BufferStore(update_buffers_[0], stored_values[0], update_indices_[0]);
    }

    // Case 3. In case the reduction is for multiple buffers, we should create the reduction with
    // LetStmt so that the reduction execution generates correct results.
    Array<Var> let_vars;
    let_vars.reserve(n_buffers_);
    for (int i = 0; i < n_buffers_; ++i) {
      Var var("v_" + update_buffers_[i]->name, PrimType(stored_values[i]->dtype));
      let_vars.push_back(var);
      buf_stores.push_back(BufferStore(update_buffers_[i], var, update_indices_[i]));
    }
    Stmt body = SeqStmt(buf_stores);
    for (int i = n_buffers_ - 1; i >= 0; --i) {
      body = LetStmt(let_vars[i], stored_values[i], std::move(body));
    }
    return body;
  }

  Optional<Stmt> CreateBlockInit(bool has_reduce_iter) {
    if (!has_reduce_iter) {
      return NullOpt;
    }

    Array<Stmt> inits;
    inits.reserve(n_buffers_);
    for (int i = 0; i < n_buffers_; ++i) {
      inits.push_back(
          BufferStore(update_buffers_[i], reducer_->identity_element[i], update_indices_[i]));
    }
    return n_buffers_ > 1 ? SeqStmt(inits) : inits[0];
  }

 public:
  /*! \brief The indices used to access the intermediate rfactor buffer */
  Array<PrimExpr> rf_buf_access_indices_;

 protected:
  /*! \brief The old block-realize */
  BlockRealize old_block_realize_;
  /*! \brief The number of block iters in the old block */
  int n_block_iters_;
  /*! \brief The rfactor loop */
  For rf_loop_;
  /*! \brief The update BufferStores of the old block */
  Array<BufferStore> old_reduction_updates_;
  /*! \brief The matched commutative reducer */
  CommReducer reducer_;
  /*! \brief The intermediate rfactor buffers */
  Array<Buffer> rf_buffers_;
  /*! \brief The number of rfactor buffers. */
  const int n_buffers_;
  /*!
   * \brief A mapping which maps old block iters to new expressions. The old iters will be replaced
   * by the expressions in future substitution for the two blocks
   */
  Map<Var, PrimExpr> var_map_;

  /*! \brief Whether we are creating the rfactor block or the write-back block */
  bool is_rf_block_;
  /*! \brief The new block iters of the new created block */
  std::vector<IterVar> iter_vars_;
  /*! \brief The new block iter bindings of the new created block-realize */
  std::vector<PrimExpr> iter_values_;
  /*! \brief The buffers updated in this block */
  Array<Buffer> update_buffers_;
  /*! \brief The indices of the buffers updated in this block, respectively */
  Array<Array<PrimExpr>> update_indices_;
  /*! \brief The LHS values of the reduction in this block */
  Array<PrimExpr> update_lhs_;
  /*! \brief THe RHS values of the reduction in this block */
  Array<PrimExpr> update_rhs_;
  /*! \brief The read regions of the new created block */
  Array<BufferRegion> read_regions_;
  /*! \brief The write regions of the new created block */
  Array<BufferRegion> write_regions_;
};

/*!
 * \brief The derived class of the rfactor block creator, which implements all virtual methods in
 * the base creator
 * \details Start constructing the rfactor block. The main difficulty to construct the rfactor block
 * is to create its block iters. So here we introduce the algorithm to create the block iters.
 *  1. Create a block iter for the rfactor loop. The block binding of this iter is the loop var, and
 *     the block iter is data parallel.
 *  2. For all the old block's block iters, there are two cases:
 *    (a) If it is data parallel block iter, or a reduction block iter which doesn't touch the
 *        rfactor loop, we keep it and its block binding in the rfactor block.
 *    (b) Otherwise it is a reduction block iter which touches the rfactor loop. In this case, we
 *        "split" the block iter into one or more new block iters and do not keep the old block
 *        var. More specifically, we create a new reduction block iter for each loop var that
 *        appears in the reduction block iter's binding (except for the rfactor loop), and the
 *        binding of the new block iter is exactly the loop var. (Note that for each loop var, we
 *        create at most one block iter, even if there are multiple old block iters which touch
 *        both this loop and the rfactor loop).
 *        Then we substitute the appearances of the old block iter with the new created block
 *        iters by recording two mappings: one maps loops vars to new created block iters which
 *        is used for binding substitution, and another maps old block iters to new expressions
 *        which is used for substitutions of the old block iters.
 */
class RFactorBlockCreator : public BaseBlockCreator {
 public:
  explicit RFactorBlockCreator(BlockRealize old_block_realize, For rf_loop,
                               Array<BufferStore> old_reduction_updates, CommReducer reducer,
                               Array<Buffer> rf_buffers,
                               std::unordered_map<const VarNode*, For> loop_vars2loop,
                               int factor_axis, Array<PrimExpr> combiner_rhs)
      : BaseBlockCreator(std::move(old_block_realize), std::move(rf_loop),
                         std::move(old_reduction_updates), std::move(reducer),
                         std::move(rf_buffers), true),
        loop_vars2loop_(std::move(loop_vars2loop)),
        factor_axis_(factor_axis),
        combiner_rhs_(std::move(combiner_rhs)) {}

 protected:
  void CreateAdditionalIter() final {
    // Create a new data parallel block iter for the rfactor loop.
    additional_iter_ =
        IterVarFromLoop(rf_loop_, "v" + rf_loop_->loop_var->name_hint, IterVarType::kDataPar);
    loop_var2block_binding_[rf_loop_->loop_var.get()] = additional_iter_->var;
    iter_vars_.push_back(additional_iter_);
    iter_values_.push_back(rf_loop_->loop_var);
  }

  void CreateNormalIters(int idx) final {
    IterVar old_iter = old_block_realize_->block->iter_vars[idx];
    PrimExpr old_binding = old_block_realize_->iter_values[idx];
    if (old_iter->iter_type == IterVarType::kDataPar ||
        !UsesVar(old_binding,
                 [v = rf_loop_->loop_var.get()](const VarNode* var) { return var == v; })) {
      // The old block iter is either a data parallel block iter, or a reduction block iter that
      // doesn't touch the rfactor loop. In this case reuse the old reduction block iter and its
      // corresponding binding.
      iter_vars_.push_back(old_iter);
      iter_values_.push_back(old_binding);
      return;
    }
    ICHECK(old_iter->iter_type == kCommReduce);
    // This block iter is a reduction block iter that touches the rfactor loop. So next we try to
    // create a new block iter for all loop vars that appear in the old binding.
    Array<Var> vars_in_old_binding = UndefinedVars(old_binding);
    for (const Var& var : vars_in_old_binding) {
      auto it = loop_vars2loop_.find(var.get());
      if (it == loop_vars2loop_.end()) {
        // `var` is not a loop var. So skip.
        continue;
      }
      const For& loop = it->second;
      if (loop_var2block_binding_.find(var.get()) == loop_var2block_binding_.end()) {
        // We haven't created the new block iter for `var`. So here we create it, append it
        // and its binding to `rf_block_iter_vars` and `rf_block_iter_values` respectively.
        IterVar new_iter_var =
            IterVarFromLoop(loop, "v" + loop->loop_var->name_hint, IterVarType::kCommReduce);
        loop_var2block_binding_[var.get()] = new_iter_var->var;
        iter_vars_.push_back(new_iter_var);
        iter_values_.push_back(var);
      }
    }
    // Substitute the original binding with new block iters. Store the result expression
    // in `rf_var_map` for future substitution.
    var_map_.Set(old_iter->var, Substitute(old_binding, loop_var2block_binding_));
  }

  void PreProcess() override {
    // The accessed indices for all reduction buffers are the same.
    rf_buf_access_indices_ = old_reduction_updates_[0]->indices;
    rf_buf_access_indices_.insert(rf_buf_access_indices_.begin() + factor_axis_,
                                  additional_iter_->var);
    for (int i = 0; i < n_buffers_; ++i) {
      update_buffers_.push_back(rf_buffers_[i]);
      update_indices_.push_back(rf_buf_access_indices_);
      update_lhs_.push_back(BufferLoad(update_buffers_[i], rf_buf_access_indices_));
      update_rhs_.push_back(combiner_rhs_[i]);
    }
  }

  void CreateReadWriteRegions() override {
    Map<Buffer, Buffer> buffer_map;
    for (int i = 0; i < n_buffers_; ++i) {
      buffer_map.Set(old_reduction_updates_[i]->buffer, rf_buffers_[i]);
    }
    const Block& old_block = old_block_realize_->block;
    read_regions_.reserve(old_block->reads.size());
    for (const BufferRegion& read_region : old_block->reads) {
      read_regions_.push_back(
          BufferRegion(read_region->buffer, Substitute(read_region->region, var_map_)));
    }
    write_regions_.reserve(old_block->writes.size());
    for (const BufferRegion& write_region : old_block->writes) {
      Array<Range> region = write_region->region;
      region.insert(region.begin() + factor_axis_,
                    Range::FromMinExtent(additional_iter_->var,
                                         make_const(additional_iter_->var.dtype(), 1)));
      Optional<Buffer> rf_buffer = buffer_map.Get(write_region->buffer);
      ICHECK(rf_buffer.defined());
      write_regions_.push_back(BufferRegion(rf_buffer.value(), Substitute(region, var_map_)));
    }
  }

 public:
  /*! \brief The generated additional block iter in rfactor block for the rfactor loop */
  IterVar additional_iter_;

 protected:
  /*!
   * \brief A mapping which maps a loop var to its corresponding For loop for all the reduction
   * block's outer loops
   */
  std::unordered_map<const VarNode*, For> loop_vars2loop_;
  /*! \brief The factor_axis specified for rfactor */
  int factor_axis_;
  /*! \brief The RHS values of the reduction in the old block */
  Array<PrimExpr> combiner_rhs_;
  /*!
   * \brief A mapping which maps loop vars to new created block iters. This map is used to
   * substitute the loop vars which appear in the bindings of some old block iters with the new
   * created block iters
   */
  std::unordered_map<const VarNode*, Var> loop_var2block_binding_;
};

Array<BufferRegion> CreateRegion(const Array<PrimExpr>& buf_loads) {
  Array<BufferRegion> buf_regions;
  for (const PrimExpr& expr : buf_loads) {
    const auto* buf_load = expr.as<BufferLoadNode>();
    ICHECK(buf_load != nullptr);
    Array<Range> region;
    region.reserve(buf_load->indices.size());
    for (const PrimExpr& index : buf_load->indices) {
      region.push_back(Range::FromMinExtent(index, make_const(index.dtype(), 1)));
    }
    buf_regions.push_back(BufferRegion(buf_load->buffer, std::move(region)));
  }
  return buf_regions;
}

/*!
 * \brief The derived class of the write-back block creator, which implements all virtual methods in
 * the base creator
 */
class WriteBackBlockCreator : public BaseBlockCreator {
 public:
  explicit WriteBackBlockCreator(BlockRealize old_block_realize, For rf_loop,
                                 Array<BufferStore> old_reduction_updates, CommReducer reducer,
                                 Array<Buffer> rf_buffers, IterVar rf_additional_iter,
                                 Array<PrimExpr> combiner_lhs,
                                 Array<PrimExpr> rf_buf_access_indices)
      : BaseBlockCreator(std::move(old_block_realize), std::move(rf_loop),
                         std::move(old_reduction_updates), std::move(reducer),
                         std::move(rf_buffers), false),
        rf_additional_iter_(std::move(rf_additional_iter)),
        combiner_lhs_(std::move(combiner_lhs)) {
    iter_vars_.reserve(n_block_iters_);
    iter_values_.reserve(n_block_iters_);
    rf_buf_access_indices_ = std::move(rf_buf_access_indices);
  }

 protected:
  void CreateAdditionalIter() override {
    // Create a new reduction block iter for the rfactor loop.
    IterVar wb_new_block_iter =
        IterVarFromLoop(rf_loop_, "v" + rf_loop_->loop_var->name_hint, kCommReduce);
    iter_vars_.push_back(wb_new_block_iter);
    iter_values_.push_back(rf_loop_->loop_var);
    var_map_.Set(rf_additional_iter_->var, wb_new_block_iter->var);
  }

  void CreateNormalIters(int idx) final {
    IterVar old_block_iter = old_block_realize_->block->iter_vars[idx];
    if (old_block_iter->iter_type == IterVarType::kDataPar) {
      iter_vars_.emplace_back(old_block_iter->dom, old_block_iter->var.copy_with_suffix(""),
                              kDataPar);
      iter_values_.push_back(old_block_realize_->iter_values[idx]);
      var_map_.Set(old_block_iter->var, iter_vars_.back());
    }
  }

  void PreProcess() override {
    for (int i = 0; i < n_buffers_; ++i) {
      PrimExpr rhs = BufferLoad(rf_buffers_[i], rf_buf_access_indices_);
      update_buffers_.push_back(old_reduction_updates_[i]->buffer);
      update_indices_.push_back(old_reduction_updates_[i]->indices);
      update_lhs_.push_back(Substitute(combiner_lhs_[i], var_map_));
      update_rhs_.push_back(Substitute(std::move(rhs), var_map_));
    }
  }

  void CreateReadWriteRegions() override {
    read_regions_ = CreateRegion(update_rhs_);
    write_regions_ = CreateRegion(update_lhs_);
  }

 protected:
  /*! \brief The new created additional block iter of the rfactor block */
  IterVar rf_additional_iter_;
  /*! \brief The LHS values of the reduction in the old block */
  Array<PrimExpr> combiner_lhs_;
};

/*!
 * \brief Create new outer loops for the rfactor block, meanwhile update the rfactor block's iter
 * bindings to use the new created loop vars
 * \param rf_block_realize The BlockRealize of the rfactor block
 * \param loops The loops to be wrapped over the rfactor block
 * \return A Stmt which is the wrapping result
 */
Stmt CreateLoopOutsideRfactorBlock(BlockRealize rf_block_realize, const Array<For>& loops) {
  int n_loops = static_cast<int>(loops.size());

  // Step 1. Create new loop vars.
  Array<For> new_loops;
  std::unordered_map<const VarNode*, Var> new_loop_var_map;
  new_loops.reserve(n_loops);
  new_loop_var_map.reserve(n_loops);
  for (const For& old_loop : loops) {
    Var new_loop_var = old_loop->loop_var.copy_with_suffix("");
    new_loop_var_map[old_loop->loop_var.get()] = new_loop_var;
  }

  // Step 2. Update the iter bindings and predicate of the rfactor block.
  Array<PrimExpr> new_bindings;
  new_bindings.reserve(rf_block_realize->iter_values.size());
  for (const PrimExpr& old_binding : rf_block_realize->iter_values) {
    new_bindings.push_back(Substitute(old_binding, new_loop_var_map));
  }
  {
    BlockRealizeNode* p_rf_block_realize = rf_block_realize.CopyOnWrite();
    p_rf_block_realize->iter_values = new_bindings;
    p_rf_block_realize->predicate = Substitute(rf_block_realize->predicate, new_loop_var_map);
  }

  // Step 3. Wrap `rf_block_realize` with outer loops.
  Stmt rf_body = rf_block_realize;
  for (int i = n_loops - 1; i >= 0; --i) {
    ObjectPtr<ForNode> p_loop = make_object<ForNode>(*loops[i].get());
    p_loop->loop_var = Downcast<Var>(new_loop_var_map[loops[i]->loop_var.get()]);
    p_loop->body = rf_body;
    rf_body = For(std::move(p_loop));
  }

  return rf_body;
}

class BlockReplacer : public StmtMutator {
 public:
  /*!
   * \brief The replace takes the old scope root block as input, and does four things:
   *  1) replace the reduction block with the write-back block,
   *  2) remove loops outside the write-back block that are touched by reduction block iters, except
   *  for the rfactor loop
   *  3) combine the rfactor block (wrapped with outer loops) and the transformed outermost loop
   *  into a SeqStmt, and
   *  4) insert the rfactor buffer into the scope root block's `alloc_buffers`
   * After transformation, the function returns the new scope root block
   * \param scope_root_block The old scope root block
   * \param rf_body The rfactor block, which is already wrapped with outer loops
   * \param outermost_loop The loop that is outermost among all loops outside the reduction block
   * \param wb_block_realize The new created BlockRealize of the write-back block
   * \param old_block_realize The BlockRealize of the reduction block
   * \param rf_loop The rfactor loop, which should be kept outside the write-back block
   * \param reduce_loop_vars The loops that are touched by reduction block iters, used to remove
   * loops outside the write-back block
   * \param loop_vars2loop The mapping from loop vars to loops that are outside the reduction block,
   * which is used to reduce redundant recursive visits
   * \param rf_buffer The rfactor buffer to be added into the scope root's `alloc_buffers`
   * \return The transformed new scope root block
   */
  static Block Replace(Block scope_root_block, Stmt rf_body, For outermost_loop,
                       BlockRealize wb_block_realize, BlockRealize old_block_realize, For rf_loop,
                       std::unordered_set<const VarNode*> reduce_loop_vars,
                       std::unordered_map<const VarNode*, For> loop_vars2loop,
                       const Array<Buffer>& rf_buffers) {
    BlockReplacer replacer(std::move(rf_body), std::move(outermost_loop),
                           std::move(wb_block_realize), std::move(old_block_realize),
                           std::move(rf_loop), std::move(reduce_loop_vars),
                           std::move(loop_vars2loop));
    Block new_scope_root = Downcast<Block>(replacer(std::move(scope_root_block)));
    BlockNode* p = new_scope_root.CopyOnWrite();
    for (const Buffer& rf_buffer : rf_buffers) {
      p->alloc_buffers.push_back(rf_buffer);
    }
    return new_scope_root;
  }

 private:
  explicit BlockReplacer(Stmt rf_body, For join_point, BlockRealize wb_block_realize,
                         BlockRealize old_block_realize, For rf_loop,
                         std::unordered_set<const VarNode*> reduce_loop_vars,
                         std::unordered_map<const VarNode*, For> loop_vars2loop)
      : rf_body_(std::move(rf_body)),
        join_point_(std::move(join_point)),
        wb_block_realize_(std::move(wb_block_realize)),
        old_block_realize_(std::move(old_block_realize)),
        rf_loop_(std::move(rf_loop)),
        reduce_loop_vars_(std::move(reduce_loop_vars)),
        loop_vars2loop_(std::move(loop_vars2loop)) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    // Step 1. Check whether this loop is outside the reduction block. Given that we've made sure
    // that the scope root block has stage-pipeline property, if this loop is not outside the
    // reduction block, there's no need to recursively mutate.
    if (!loop_vars2loop_.count(loop->loop_var.get())) {
      return GetRef<For>(loop);
    }

    // Step 2. Recursively mutate.
    Stmt body = StmtMutator::VisitStmt(loop->body);

    // Step 3. If this loop is the rfactor loop and isn't touched by any reduction block iter, it
    // should be kept outside the write-back block. Otherwise it shouldn't.
    if (loop == rf_loop_.get() || !reduce_loop_vars_.count(loop->loop_var.get())) {
      ObjectPtr<ForNode> p_loop = CopyOnWrite(loop);
      p_loop->body = body;
      body = Stmt(p_loop);
    }

    // Step 4. If we're at the specified join point, return the combination of
    // `rf_body_` and the mutation result `body`. Otherwise return the mutation result.
    return loop == join_point_.get() ? SeqStmt({rf_body_, body}) : body;
  }

  Stmt VisitStmt_(const BlockRealizeNode* block_realize) final {
    // Replace old_block_realize_ with wb_block_realize_.
    if (block_realize == old_block_realize_.get()) {
      return wb_block_realize_;
    }
    return StmtMutator::VisitStmt_(block_realize);
  }

  Stmt VisitStmt_(const SeqStmtNode* seq) final {
    Array<Stmt> new_stmts;
    new_stmts.reserve(static_cast<int>(seq->seq.size()));

    for (const Stmt old_stmt : seq->seq) {
      new_stmts.push_back(VisitStmt(old_stmt));
    }
    return SeqStmt::Flatten(new_stmts);
  }

 private:
  Stmt rf_body_;
  For join_point_;
  BlockRealize wb_block_realize_;
  BlockRealize old_block_realize_;
  For rf_loop_;
  std::unordered_set<const VarNode*> reduce_loop_vars_;
  std::unordered_map<const VarNode*, For> loop_vars2loop_;
};

StmtSRef RFactor(ScheduleState self, const StmtSRef& rf_loop_sref, int factor_axis,
                 bool merge_loops) {
  // *****************************************************
  // *    Condition Checks and Information Collection    *
  // *****************************************************

  // Step 1. Check some basic conditions for rfactor. Get the block and block-realize.
  BlockRealize block_realize = CheckGetSingleChildBlockRealizeOnSRefTree(self, rf_loop_sref);
  const StmtSRef& block_sref = self->stmt2ref.at(block_realize->block.get());
  const Block& block = block_realize->block;
  StmtSRef scope_root = GetScopeRoot(self, block_sref,  //
                                     /*require_stage_pipeline=*/true);
  if (self->enable_check) {
    CheckReductionBlock(self, block_sref, scope_root);
  }
  auto rf_loop = GetRef<For>(TVM_SREF_TO_FOR(rf_loop_sref));
  if (rf_loop->kind != ForKind::kSerial) {
    throw NotSerialLoopKindError(self->mod, rf_loop);
  }

  // Step 2. Collect loop vars that are touched by data parallel block iters and reduction block
  // iters, respectively.
  std::unordered_set<const VarNode*> data_par_loop_vars;
  std::unordered_set<const VarNode*> reduce_loop_vars;
  GetVarsTouchedByBlockIters(block_realize, &data_par_loop_vars, &reduce_loop_vars);

  // Step 3. Collect the loops of the reduction block. Construct a mapping from loops to
  // corresponding loop vars.
  Array<For> loops = LoopSRefs2Loops(GetLoops(block_sref));
  std::unordered_map<const VarNode*, For> loop_vars2loop = GetLoopVar2LoopMap(loops);

  // Step 4. Check four properties that the loops should have:
  // - the rfactor loop cannot be touched by any data parallel block iter;
  // - all the loops cannot be touched by both data parallel block iters and reduction block iters;
  // - the outermost loop should have the reduction block as its first child block;
  // - the outermost loop that is touched by some reduction block iters can only have one child
  // block.
  if (self->enable_check) {
    LoopPropertyError::CheckLoopProperty(self, loops, rf_loop.get(), block, data_par_loop_vars,
                                         reduce_loop_vars);
  }

  // Step 4.1. To merge loop nests (if `merge_loops == true`), only keep the loops that are inside
  // the reduction loop (excluding the reduction loop itself).
  if (merge_loops) {
    auto it = std::find(loops.begin(), loops.end(), rf_loop);
    ICHECK(it != loops.end()) << "The rfactor loop is not found in the loops";
    loops.erase(loops.begin(), it + 1);
  }

  // Step 5. Get the `init` identity and the `update` combiner of the reduction. Extract the
  // commutative reducer, combiner lhs and combiner rhs from the reduction identity and the
  // reduction combiner. The lhs will be used when constructing the write-back block, and the rhs
  // will be used when constructing the rfactor block.
  Array<PrimExpr> init_values{nullptr};
  Array<BufferStore> updates{nullptr};
  CommReducer reducer{nullptr};
  Array<PrimExpr> combiner_lhs{nullptr};
  Array<PrimExpr> combiner_rhs{nullptr};
  std::tie(init_values, updates) = GetInitValuesAndUpdatesFromReductionBlock(self, block);
  std::tie(reducer, combiner_lhs, combiner_rhs) =
      GetReducerAndCombinerLhsRhs(self, init_values, updates);

  // Step 6. Check whether `factor_axis` is in a correct range, and convert it to non-negative if it
  // is negative.
  factor_axis =
      FactorAxisOutOfRangeError::CheckAndUpdate(self->mod, updates[0]->buffer, factor_axis);

  // *****************************************************
  // *                 IR Manipulation                   *
  // *****************************************************
  // Since rfactor splits the reduction block into two, we call the first one "rfactor block", and
  // the latter one "write-back block", and the intermediate buffer is called "rfactor buffer".

  // Step 1. Create the intermediate buffer (a.k.a. rfactor buffer), which has an additional
  // dimension that specified by `factor_axis` and `rf_loop`.
  Array<Buffer> rf_buffers = CreateExpandedBuffers(updates, factor_axis, rf_loop->extent);

  // Step 2. Create the rfactor block.
  RFactorBlockCreator rf_block_creator(block_realize, rf_loop, updates, reducer, rf_buffers,
                                       loop_vars2loop, factor_axis, std::move(combiner_rhs));
  auto rf_br = rf_block_creator.CreateBlock();

  // Step 3. Create the write-back block.
  WriteBackBlockCreator wb_block_creator(block_realize, rf_loop, updates, reducer, rf_buffers,
                                         std::move(rf_block_creator.additional_iter_),
                                         std::move(combiner_lhs),
                                         std::move(rf_block_creator.rf_buf_access_indices_));
  auto wb_br = wb_block_creator.CreateBlock();

  // Step 4. Wrap the rfactor block with loops.
  Stmt rf_body = CreateLoopOutsideRfactorBlock(rf_br, loops);

  // *****************************************************
  // *           Schedule Replacement & Update           *
  // *****************************************************

  // Step 1. Substitute the old scope root block with the new scope root block.
  Block old_scope_root_block = GetRef<Block>(scope_root->StmtAs<BlockNode>());
  Block new_scope_root_block =
      BlockReplacer::Replace(old_scope_root_block, rf_body, loops[0], wb_br, block_realize, rf_loop,
                             reduce_loop_vars, loop_vars2loop, rf_buffers);
  self->Replace(scope_root, new_scope_root_block,
                {{old_scope_root_block, new_scope_root_block}, {block, wb_br->block}});

  // Step 2. Update scope information.
  std::vector<StmtSRef> new_block_srefs{self->stmt2ref.at(rf_br->block.get()),
                                        self->stmt2ref.at(wb_br->block.get())};
  for (const StmtSRef& new_block_sref : new_block_srefs) {
    BlockInfo& info = self->block_info[new_block_sref];
    info.affine_binding = true;
    info.region_cover = true;
    info.stage_pipeline = true;
  }
  return new_block_srefs[0];
}

/* Rolling Update, Split-K Update */

namespace {

template <typename K, typename V>
using ObjectMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;
template <typename K>
using ObjectSet = std::unordered_set<K, ObjectPtrHash, ObjectPtrEqual>;

struct ReduceBlockFrontier {
  virtual bool HasIncompleteBuffer(const Buffer& buffer) const = 0;
};

struct RollingUpdateBlockFrontier : ReduceBlockFrontier {
  static std::tuple<RollingUpdateBlockFrontier, std::vector<StmtSRef>> Build(
      const ScheduleState& self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
      const StmtSRef& scope_root_sref, const size_t scan_dim);

  void ApplyAllReduceToScan(ScheduleState& self, const StmtSRef& loop_sref);

  std::optional<size_t> GetScanDim(const Buffer& buffer) const {
    auto it = bufs_scan_dim_.find(buffer);
    return it == bufs_scan_dim_.end() ? std::optional<size_t>() : it->second;
  }

  bool HasIncompleteBuffer(const Buffer& buffer) const override {
    return GetScanDim(buffer).has_value();
  }

 private:
  RollingUpdateBlockFrontier(ObjectSet<StmtSRef> block_srefs, size_t arg_scan_dim)
      : arg_scan_dim_(arg_scan_dim), block_srefs_(std::move(block_srefs)) {}

  size_t arg_scan_dim_;
  ObjectSet<StmtSRef> block_srefs_;
  // This one is populated later by `ApplyAllReduceToScan`.
  ObjectMap<Buffer, size_t> bufs_scan_dim_;
};

struct SplitKBlockFrontier : ReduceBlockFrontier {
  static std::tuple<SplitKBlockFrontier, std::vector<StmtSRef>> Build(
      const ScheduleState& self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
      const StmtSRef& scope_root_sref);

  bool InitiateBlockBufferSubst(ScheduleState& self, const StmtSRef& block_sref,
                                const StmtSRef& loop_sref) const;
  void FinishBlockBufferSubst(ScheduleState& self, const StmtSRef& block_sref,
                              const StmtSRef& loop_sref) const;

  auto GetWBBufferFromRFBuffer(const Buffer& buffer) const {
    auto it = rf_to_wb_.find(buffer);
    return it == rf_to_wb_.end() ? std::optional<std::pair<Buffer, size_t>>() : it->second;
  }

  bool HasIncompleteBuffer(const Buffer& buffer) const override {
    return GetWBBufferFromRFBuffer(buffer).has_value();
  }

 private:
  SplitKBlockFrontier(const ObjectSet<StmtSRef>& post_l_frontier,
                      const ObjectSet<StmtSRef>& under_l_frontier);

  ObjectMap<Buffer, std::pair<Buffer, size_t>> wb_to_rf_, rf_to_wb_;
};

struct ReduceRepairer {
  static ReduceRepairer DeriveFromBlockExpr(const CommReducer& reducer, PrimExpr combiner_rhs,
                                            const ReduceBlockFrontier& frontier);

  using BufLoadRewriter = std::function<std::pair<BufferLoad, BufferLoad>(const BufferLoad&)>;

  Block RewriteBlockExpr(Block block, const Map<Var, PrimExpr>& iv_transform,
                         BufLoadRewriter buf_load_rewriter, PrimExpr load_condition,
                         arith::Analyzer& analyzer, bool apply_to_lhs) const;

 private:
  using BufAndVxT = std::vector<std::tuple<BufferLoad, Var, Var>>;

  ReduceRepairer(PrimExpr h_expr, Var vr, BufAndVxT buf_and_vx)
      : h_expr_(std::move(h_expr)), vr_(std::move(vr)), buf_and_vx_(std::move(buf_and_vx)) {}

  ReduceRepairer() = default;

 public:
  PrimExpr h_expr_{};
  Var vr_{};
  BufAndVxT buf_and_vx_{};
};

/*!
 \brief Mock-inline all the blocks in `blocks`. Requires `blocks` to be topologically sorted where
 `blocks[-1]` is the last block. The inlining is done in a new ScheduleState, without modifying
 `_cur_state`.
 \return The new version of the last block after inlining.
 */
Block MockInlineBlocks(const ScheduleState& _cur_state, const std::vector<StmtSRef>& blocks);

struct RFactorInternalProducts {
  Buffer rf_buffer;
  StmtSRef rf_block_sref;
  WriteBackBlockCreator wb_block_creator;
};

/*!
 \brief Run \ref RFactor, and return more intermediate products in RFactorInternalProducts.
*/
RFactorInternalProducts RFactorInternal(ScheduleState self, const StmtSRef& scope_root_sref,
                                        const StmtSRef& block_sref, const StmtSRef& loop_sref,
                                        int factor_axis, bool add_annotations);

}  // namespace

StmtSRef RollingUpdate(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                       int factor_axis) {
  // See the docstring of `RollingUpdate` for a high-level overview. It describes what `b0`, `Br`,
  // `Bt` are, and these symbols are important here. Here we describe the details as we go:

  // Step 1. Find the reduce producers of `b0` as `Br`, and produce the topological order `Bt`.
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  auto [frontier, topo_order] =
      RollingUpdateBlockFrontier::Build(self, block_sref, loop_sref, scope_root_sref, factor_axis);
  // Step 2. Reverse-compute-at all the blocks in `Bt` into the loop `l`.
  for (auto& block_sref : topo_order) {
    // ...only if `block_sref` is not already under `loop_sref`.
    if (!GetSRefLowestCommonAncestor({block_sref, loop_sref}).same_as(loop_sref)) {
      tir::UnsafeReverseComputeAt(self, block_sref, loop_sref, /*preserve_unit_loops=*/true);
    }
  }
  // Step 3. Apply `ReduceToScan` to convert each reduction block in `Br` into a scan block.
  frontier.ApplyAllReduceToScan(self, loop_sref);

  // Step 4. Algebraic rewrite starts. Mock-inline all of `Bt` into `b0`, which produces a `b0_new`
  // with all the computation concentrated into it. Match `b0_new` against TVM-known reduction
  // patterns, then produce a `ReduceRepairer` function.
  arith::Analyzer analyzer;
  auto new_b0 = MockInlineBlocks(self, topo_order);
  auto red_match = tir::MatchSelfReduction(self, new_b0, std::nullopt);
  auto repairer = ReduceRepairer::DeriveFromBlockExpr(red_match.reducer,
                                                      analyzer.Simplify(red_match.rhs), frontier);

  // Step 5. Run RFactor to factorize `b0` over `l`.
  auto rfactor_result = RFactorInternal(self, scope_root_sref, block_sref, loop_sref, factor_axis,
                                        /*add_annotations=*/false);

  // Step 6. Apply the `ReduceRepairer` function to the write-back block.
  auto wb_block = GetRef<Block>(TVM_SREF_TO_BLOCK(block_sref));
  // NOTE: we're assuming the first iter-var is the RFactor iter-var. See
  // `WriteBackBlockCreator`. If the logic changes there, we need to change this too.
  Var rf_iv_var = wb_block->iter_vars[0]->var;
  // Create a substitution map from the x-vars, x'-vars, and r-var to their expressions.
  auto iv_var_map = rfactor_result.wb_block_creator.GetVarMap();
  Block new_wb_block = repairer.RewriteBlockExpr(
      wb_block, iv_var_map,
      [&, &frontier = frontier](const BufferLoad& buf_load) {
        size_t rf_dim = frontier.GetScanDim(buf_load->buffer).value();
        auto prev_buf_load = buf_load;
        prev_buf_load.CopyOnWrite()->indices.Set(rf_dim, rf_iv_var - 1);
        auto curr_buf_load = buf_load;
        curr_buf_load.CopyOnWrite()->indices.Set(rf_dim, rf_iv_var);
        return std::make_pair(prev_buf_load, curr_buf_load);
      },
      // Since we create a BufferLoad that uses `rf_iv_var - 1`, we need to put a Select condition
      // around the load.
      /*load_condition=*/rf_iv_var > 0, analyzer, /*apply_to_lhs=*/true);
  self->Replace(block_sref, new_wb_block, {{wb_block, new_wb_block}});
  return rfactor_result.rf_block_sref;
}

StmtSRef SplitKUpdate(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                      int factor_axis) {
  // See the docstring of `SplitKUpdate` for a high-level overview. It describes what `b0`, `Br`,
  // `Bt` are, and these symbols are important here. Also see the docstring of `RollingUpdate` for a
  // comparison. Here we describe the details as we go:

  // Step 1. Find the reduce producers of `b0` as `Br`, and produce the topological order `Bt`.
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  auto [frontier, topo_order] =
      SplitKBlockFrontier::Build(self, block_sref, loop_sref, scope_root_sref);
  // Step 2. Substitute the buffer reads of each block in `Bt`, and fuse them under `l`.
  std::vector<bool> block_modified(topo_order.size(), false);
  for (size_t i = 0; i < topo_order.size(); ++i) {
    block_modified[i] = frontier.InitiateBlockBufferSubst(self, topo_order[i], loop_sref);
  }
  // NOTE: the use-def relation of the blocks has changed, so we need to update it.
  // You may think that `self->Replace` should do it, but it doesn't seem to, maybe because
  // modifications to the computation expression in blocks are rare in TVM.
  // To make sure `UpdateScopeBlockInfo` works, it's important that we start from the BlockRealize
  // of the scope root, not the scope root block itself.
  self->UpdateScopeBlockInfo(GetBlockRealize(self, scope_root_sref));
  for (size_t i = 0; i < topo_order.size(); ++i) {
    if (!GetSRefLowestCommonAncestor({topo_order[i], loop_sref}).same_as(loop_sref)) {
      tir::UnsafeReverseComputeAt(self, topo_order[i], loop_sref, /*preserve_unit_loops=*/true);
    }
    if (block_modified[i]) {
      frontier.FinishBlockBufferSubst(self, topo_order[i], loop_sref);
    }
  }

  // Step 3. Algebraic rewrite starts. Mock-inline all of `Bt` into `b0`, which produces a `b0_new`
  // with all the computation concentrated into it. Match `b0_new` against TVM-known reduction
  // patterns, then produce a `ReduceRepairer` function.
  arith::Analyzer analyzer;
  auto new_block = MockInlineBlocks(self, topo_order);
  auto new_red_match = tir::MatchSelfReduction(self, new_block, std::nullopt);
  auto repairer = ReduceRepairer::DeriveFromBlockExpr(
      new_red_match.reducer, analyzer.Simplify(new_red_match.rhs), frontier);

  // Step 4. Run RFactor to factorize `b0` over `l`.
  auto rfactor_result = RFactorInternal(self, scope_root_sref, block_sref, loop_sref, factor_axis,
                                        /*add_annotations=*/true);

  // Step 5. Apply the `ReduceRepairer` function to the write-back block.
  auto wb_block = GetRef<Block>(TVM_SREF_TO_BLOCK(block_sref));
  Var rf_iv_var = wb_block->iter_vars[0]->var;
  auto iv_var_map = rfactor_result.wb_block_creator.GetVarMap();
  Block new_wb_block = repairer.RewriteBlockExpr(
      wb_block, iv_var_map,
      [&, &frontier = frontier](const BufferLoad& buf_load) {
        auto [wb_buffer, rf_dim] = frontier.GetWBBufferFromRFBuffer(buf_load->buffer).value();
        auto prev_buf_load = buf_load;
        prev_buf_load.CopyOnWrite()->indices.Set(rf_dim, rf_iv_var);
        auto curr_load_indices = buf_load->indices;
        curr_load_indices.erase(curr_load_indices.begin() + rf_dim);
        BufferLoad curr_buf_load(wb_buffer, curr_load_indices);
        return std::make_pair(prev_buf_load, curr_buf_load);
      },
      /*load_condition=*/PrimExpr(), analyzer, /*apply_to_lhs=*/false);
  self->Replace(block_sref, new_wb_block, {{wb_block, new_wb_block}});

  // Step 6. Pull the write-back block (now under `l`) out to root position.
  tir::ReverseComputeRoot(self, block_sref);
  return rfactor_result.rf_block_sref;
}

namespace {

BufferRegion GetUniqueWrite(const BlockNode* block) {
  ICHECK(block->writes.size() == 1) << "Only one write is supported";
  return block->writes[0];
}

bool IsIncompleteUnderLoop(const ScheduleState& self, const StmtSRef& block_sref,
                           const StmtSRef& loop_sref) {
  // 1. If `loop` is not among the loops of the block, return false.
  auto loops = GetLoops(block_sref);
  auto loop_it = std::find(loops.begin(), loops.end(), loop_sref);
  if (loop_it == loops.end()) {
    return false;
  }
  // 2. Check if any reduction iter-var uses `loop`. If so, return true.
  // If any reduction iter-var uses loops outside of `loop`, that's an error because we don't know
  // how to handle that.
  auto br = GetBlockRealize(self, block_sref);
  auto forbidden_loop_vars = support::map(loops.begin(), loop_it, [](const StmtSRef& loop_sref) {
                               return TVM_SREF_TO_FOR(loop_sref)->loop_var.get();
                             }).to_container<std::unordered_set>();
  auto loop = TVM_SREF_TO_FOR(loop_sref);
  for (size_t i = 0; i < br->iter_values.size(); ++i) {
    auto iv = br->block->iter_vars[i];
    auto iv_expr = br->iter_values[i];
    if ((iv->iter_type != IterVarType::kCommReduce && iv->iter_type != IterVarType::kOrdered)) {
      continue;
    }
    bool uses_forbidden_vars =
        UsesVar(iv_expr, [&](const VarNode* var) { return forbidden_loop_vars.count(var); });
    ICHECK(!uses_forbidden_vars)
        << "Block reduction iter-var " << iv << " = " << iv_expr
        << " uses loops outside of the target loop. This is not supported.";
    bool uses_loop_var =
        UsesVar(iv_expr, [&](const VarNode* var) { return loop->loop_var.get() == var; });
    if (uses_loop_var) {
      return true;
    }
  }
  return false;
}

struct AfterLoopFinder {
  AfterLoopFinder(const StmtSRef& scope_root_sref, const For& loop) : loop_(loop) {
    bool is_after_l = false;
    PostOrderVisit(GetRef<Stmt>(scope_root_sref->stmt), [&](const ObjectRef& node) {
      if (node.same_as(loop)) {
        is_after_l = true;
      } else if (auto block = node.as<Block>(); block && is_after_l) {
        blocks_after_loop_.insert(block.value().get());
      }
    });
  }

  bool IsReductionAfterLoop(const StmtSRef& block_sref) const {
    auto block = TVM_SREF_TO_BLOCK(block_sref);
    auto& ivs = block->iter_vars;
    bool has_reduce_iv = std::any_of(ivs.begin(), ivs.end(), [](const IterVar& iv) {
      return iv->iter_type == IterVarType::kCommReduce;
    });
    return has_reduce_iv && blocks_after_loop_.count(block);
  }

  const For& loop_;
  std::unordered_set<const BlockNode*> blocks_after_loop_;
};

std::optional<std::tuple<bool, Buffer, size_t>> PreviousRollingUpdateAnnot(const BlockNode* block) {
  auto rf_buffer_annot_ = GetAnn<Array<ObjectRef>>(block, attr::tir_rfactor_data);
  if (!rf_buffer_annot_.defined()) {
    return std::nullopt;
  }
  auto rf_buffer_annot = rf_buffer_annot_.value();
  ICHECK(rf_buffer_annot.size() == 3);
  auto is_writeback = Downcast<Bool>(rf_buffer_annot[0]);
  auto rf_buffer = Downcast<Buffer>(rf_buffer_annot[1]);
  auto factor_axis = Downcast<Integer>(rf_buffer_annot[2]);
  return std::make_tuple(is_writeback->value, rf_buffer, factor_axis->value);
}

bool IsAnnotatedRFactorBlock(const BlockNode* block) {
  auto annot = PreviousRollingUpdateAnnot(block);
  return annot.has_value() && !std::get<0>(annot.value());
}

auto FrontierBuildHelper(const ScheduleState& self, const StmtSRef& b0_sref,
                         const StmtSRef& loop_sref, const StmtSRef& scope_root_sref) {
  // Get all blocks after `loop`. We'll use it in a bit.
  auto loop = GetRef<For>(TVM_SREF_TO_FOR(loop_sref));
  AfterLoopFinder block_finder(scope_root_sref, loop);

  // NOTE:
  // 1. The "frontiers" contain producers of `b0` that are "incomplete", which means they are
  // either reductions under `loop`, which we put in `under_l_frontier`, or just outright after
  // `loop`, which we put in `post_l_frontier`. (Of course `b0` itself is also after `loop`, but we
  // don't include it in `post_l_frontier`.)
  // 2. `topo_order` only contains blocks that are on a path that leads into any of the frontier
  // blocks. It always ends in `b0`.
  // 3. Because `Visitor` needs to look at `found_incomplete`, insertion into `topo_order`
  // happens after the recursive call, which creates a reverse topological order.
  ObjectSet<StmtSRef> post_l_frontier, under_l_frontier, seen;
  std::vector<StmtSRef> topo_order;
  std::function<bool(const StmtSRef&)> Visitor = [&](const StmtSRef& block_sref) -> bool {
    seen.insert(block_sref);
    auto block = TVM_SREF_TO_BLOCK(block_sref);
    if (IsIncompleteUnderLoop(self, block_sref, loop_sref) || IsAnnotatedRFactorBlock(block)) {
      under_l_frontier.insert(block_sref);
      return true;
    } else if (!block_sref.same_as(b0_sref) && block_finder.IsReductionAfterLoop(block_sref)) {
      post_l_frontier.emplace(block_sref);
      return true;
    }
    bool found_incomplete = false;
    for (auto& producer_sref : GetProducers(self, block_sref)) {
      if (!seen.count(producer_sref)) {
        found_incomplete |= Visitor(producer_sref);
      }
    }
    if (found_incomplete) {
      topo_order.push_back(block_sref);
    }
    return found_incomplete;
  };

  Visitor(b0_sref);
  // If there's anything in `topo_order` at all, it should end in `block_sref`. However, if it's
  // empty, manually add it to satisfy the contract.
  if (topo_order.empty()) {
    topo_order.push_back(b0_sref);
  } else {
    ICHECK(topo_order.back().same_as(b0_sref));
  }
  return std::make_tuple(post_l_frontier, under_l_frontier, topo_order);
}

std::tuple<RollingUpdateBlockFrontier, std::vector<StmtSRef>> RollingUpdateBlockFrontier::Build(
    const ScheduleState& self, const StmtSRef& b0_sref, const StmtSRef& loop_sref,
    const StmtSRef& scope_root_sref, size_t scan_dim) {
  auto [post_l_frontier, under_l_frontier, topo_order] =
      FrontierBuildHelper(self, b0_sref, loop_sref, scope_root_sref);
  ICHECK(post_l_frontier.empty())
      << "Found the following post-loop producers in scan mode (only allowed in post-reduce mode): "
      << [&xs = post_l_frontier]() {
           auto GetName = [&](const StmtSRef& block_sref) {
             return TVM_SREF_TO_BLOCK(block_sref)->name_hint;
           };
           return support::map(xs, GetName).to_vector();
         }();
  return {RollingUpdateBlockFrontier(under_l_frontier, scan_dim), topo_order};
}

void RollingUpdateBlockFrontier::ApplyAllReduceToScan(ScheduleState& self,
                                                      const StmtSRef& loop_sref) {
  for (auto& block_sref : block_srefs_) {
    auto scan_dim_annot = GetAnn<Integer>(block_sref, attr::tir_scan_buf_dim);
    if (scan_dim_annot.defined()) {
      auto write = GetUniqueWrite(TVM_SREF_TO_BLOCK(block_sref));
      bufs_scan_dim_.emplace(write->buffer, scan_dim_annot.value()->value);
    } else {
      tir::ReduceToScan(self, block_sref, loop_sref, /*write_buffer_index=*/0, arg_scan_dim_);
      auto write = GetUniqueWrite(TVM_SREF_TO_BLOCK(block_sref));
      bufs_scan_dim_.emplace(write->buffer, arg_scan_dim_);
    }
  }
}

std::tuple<SplitKBlockFrontier, std::vector<StmtSRef>> SplitKBlockFrontier::Build(
    const ScheduleState& self, const StmtSRef& b0_sref, const StmtSRef& loop_sref,
    const StmtSRef& scope_root_sref) {
  auto [post_l_frontier, under_l_frontier, topo_order] =
      FrontierBuildHelper(self, b0_sref, loop_sref, scope_root_sref);
  ObjectMap<Buffer, std::pair<Buffer, size_t>> frontier_buf_wb_to_rf;
  auto AddBlockBuffer = [&](const StmtSRef& block_sref) {
    auto block = TVM_SREF_TO_BLOCK(block_sref);
    auto [is_writeback, rf_buffer, rf_buffer_dim] = PreviousRollingUpdateAnnot(block).value();
    if (is_writeback) {
      auto write = GetUniqueWrite(block);
      frontier_buf_wb_to_rf.emplace(write->buffer, std::make_pair(rf_buffer, rf_buffer_dim));
    }
  };
  for (auto& block_sref : under_l_frontier) {
    AddBlockBuffer(block_sref);
  }
  for (auto& block_sref : post_l_frontier) {
    AddBlockBuffer(block_sref);
  }
  return {SplitKBlockFrontier(post_l_frontier, under_l_frontier), topo_order};
}

SplitKBlockFrontier::SplitKBlockFrontier(const ObjectSet<StmtSRef>& post_l_frontier,
                                         const ObjectSet<StmtSRef>& under_l_frontier) {
  auto PreviousRollingUpdateAnnot_ = [](const BlockNode* block) {
    auto annot = PreviousRollingUpdateAnnot(block);
    ICHECK(annot.has_value()) << "Block " << block->name_hint << " is expected to have annotation "
                              << attr::tir_rfactor_data << " in split-k-update";
    return annot.value();
  };
  for (auto& block_sref : post_l_frontier) {
    auto block = TVM_SREF_TO_BLOCK(block_sref);
    auto [is_writeback, rf_buffer, rf_buffer_dim] = PreviousRollingUpdateAnnot_(block);
    ICHECK(is_writeback) << "Block " << block->name_hint << " is expected to be a writeback block "
                         << "in split-k-update";
    auto write = GetUniqueWrite(block);
    wb_to_rf_.emplace(write->buffer, std::make_pair(rf_buffer, rf_buffer_dim));
    rf_to_wb_.emplace(rf_buffer, std::make_pair(write->buffer, rf_buffer_dim));
  }
  for (auto& block_sref : under_l_frontier) {
    auto block = TVM_SREF_TO_BLOCK(block_sref);
    auto [is_writeback, wb_buffer, rf_buffer_dim] = PreviousRollingUpdateAnnot_(block);
    ICHECK(!is_writeback) << "Block " << block->name_hint << " is expected to be an rfactor block "
                          << "in split-k-update";
    auto write = GetUniqueWrite(block);
    rf_to_wb_.emplace(write->buffer, std::make_pair(wb_buffer, rf_buffer_dim));
  }
}

bool SplitKBlockFrontier::InitiateBlockBufferSubst(ScheduleState& self, const StmtSRef& block_sref,
                                                   const StmtSRef& loop_sref) const {
  auto block = GetRef<Block>(TVM_SREF_TO_BLOCK(block_sref));
  auto loop_var = TVM_SREF_TO_FOR(loop_sref)->loop_var;
  auto new_block =
      Downcast<Block>(ReplaceBufferLoads(block, [&](const BufferLoadNode* op) -> PrimExpr {
        auto it = wb_to_rf_.find(op->buffer);
        if (it == wb_to_rf_.end()) {
          return GetRef<PrimExpr>(op);
        } else {
          auto [rf_buffer, rf_buffer_dim] = it->second;
          auto indices = op->indices;
          indices.insert(indices.begin() + rf_buffer_dim, loop_var);
          return BufferLoad(rf_buffer, indices);
        }
      }));
  if (new_block.same_as(block)) {
    return false;
  }
  self->Replace(block_sref, new_block, {{block, new_block}});
  return true;
}

void SplitKBlockFrontier::FinishBlockBufferSubst(ScheduleState& self, const StmtSRef& block_sref,
                                                 const StmtSRef& loop_sref) const {
  // Create a new iter-var `v_j` for the loop var `j`, add `v_j` to the block's iter-vars, and
  // substitute existing uses of `j` with `v_j`.
  auto loop = GetRef<For>(TVM_SREF_TO_FOR(loop_sref));
  auto loop_var = loop->loop_var;
  MutateAndReplaceBlockRealize(self, block_sref, [&](BlockRealize br) {
    auto [new_br, new_iv] = SplitVarFromIterVars(br, loop_var, Range::FromExtent(loop->extent));
    auto block = Downcast<Block>(Substitute(new_br->block, {{loop_var, new_iv->var}}));
    return BlockRealize(new_br->iter_values, new_br->predicate, block, new_br->span);
  });
}

Block MockInlineBlocks(const ScheduleState& _cur_state, const std::vector<StmtSRef>& blocks) {
  // NOTE: because TVM does not support undoing of inlining, we will create a new ScheduleState
  // where the inlining happens. We don't clone the IRModule itself.
  // We also need to transport `blocks` to the new state.
  ScheduleState new_state(_cur_state->mod, _cur_state->debug_mask, _cur_state->enable_check);
  auto new_blocks = support::map(blocks, [&](const StmtSRef& block_sref) {
                      return new_state->stmt2ref.at(block_sref->stmt);
                    }).to_vector();
  ICHECK(!new_blocks.empty());
  // `new_blocks` is topologically sorted, and when we inline the blocks, all the other blocks
  // should get inlined into the last block (meaning that we don't want to inline the last block).
  auto last_block = new_blocks.back();
  new_blocks.pop_back();
  auto scope_root_sref = GetScopeRoot(new_state, last_block, /*require_stage_pipeline=*/false);
  for (auto& block_sref : new_blocks) {
    auto [new_scope_root, block_reuse] =
        tir::ApplyInlineAndGenerateScopeRoot(new_state, block_sref, scope_root_sref);
    new_state->Replace(scope_root_sref, new_scope_root, block_reuse);
  }
  return GetRef<Block>(TVM_SREF_TO_BLOCK(last_block));
}

struct Memoizer {
  std::unordered_map<std::string, std::string> cached_results;
  std::mutex mutex;

  static Memoizer global_memoizer;

  static std::optional<std::string> Get(const std::string& key) {
    std::lock_guard<std::mutex> lock(global_memoizer.mutex);
    auto it = global_memoizer.cached_results.find(key);
    if (it != global_memoizer.cached_results.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  static void Set(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(global_memoizer.mutex);
    global_memoizer.cached_results[key] = value;
  }
};

Memoizer Memoizer::global_memoizer;

/*!
 \brief Solves for a global updater `H` that can update the final result that has been reduced by
 the comm-reducer `f`, in the style of FlashAttention.
 \param reducer an associative, commutative reducer `f`.
 \param g_expr a map function, written as an expression of `x`-variables (see x2x).
 \param x2x a map from x-variables (old x) to x'-variables (new x).
 \param r a variable that represents the reduction result.
 \return the global updater `H`, written as an expression of `r` and `x2x` variables.
*/
PrimExpr HExprSolver(const CommReducer& reducer, const PrimExpr& g_expr, const Map<Var, Var>& x2x,
                     const Var& r) {
  // Cache the result. This function (and rolling-update, in general) may be called thousands of
  // times due to the autotuner replaying traces.
  // We save the string representation of PrimExprs, because PrimExpr itself compares by pointer
  // (and even if we use StructuralEqual, in the end Vars compare by pointer, not name).
  auto cache_key = SExprPrinter::Print(g_expr);
  if (auto value_str = Memoizer::Get(cache_key)) {
    // If the cache hit, parse the result and return it.
    std::unordered_map<std::string, Var> varmap({{r->name_hint, r}});
    for (auto& [x, x_] : x2x) {
      varmap[x->name_hint] = x;
      varmap[x_->name_hint] = x_;
    }
    return SExprParser::Parse(value_str.value(), varmap);
  }

  auto FindAndCheckFunc = [](const std::string& name) {
    auto* func = runtime::Registry::Get(name);
    ICHECK(func) << name << " is not registered.";
    return (*func);
  };
  const auto& seperator = FindAndCheckFunc("arith.sympy.separate_vars");
  const auto& inverter = FindAndCheckFunc("arith.sympy.inverse");
  const auto& simplifier = FindAndCheckFunc("arith.sympy.simplify");

  // 1. Attempt to separate `u`-variables and `x`-variables in `g`, such that `g(u..., x...) =
  // gcomb(gu(u...), gx(x...))`. If there is only one `u` and only one `x`, this step is trivial.
  Array<Var> xvars;
  for (auto& [x, _] : x2x) {
    xvars.push_back(x);
  }
  // gcomb will be given as an expression of `a` and `b`.
  Var a("a", g_expr->dtype), b("b", g_expr->dtype);
  Optional<Array<PrimExpr>> separated = seperator(g_expr, xvars, a, b);
  ICHECK(separated.defined()) << "Cannot separate the variables in the combiner RHS.";
  ICHECK(separated.value().size() == 3);
  auto gx = separated.value()[0], gcomb = separated.value()[2];
  // 2. Attempt to invert `gcomb(a, b) = r` wrt `a` to get `a = gcomb^(-1)(r, b)`.
  // gcomb_inv will be given as an expression of `r` and `b`.
  PrimExpr gcomb_inv = inverter(gcomb, a, r);
  // 3. Compute h(r, x, x') = g_comb(g_comb^(-1)(r, gx(x)), gx(x')).
  PrimExpr gcomb_inv_ = Substitute(gcomb_inv, {{b, gx}});  // gcomb^(-1)(r, gx(x))
  PrimExpr gx_prime = Substitute(gx, x2x);                 // gx(x')
  PrimExpr h = simplifier(Substitute(gcomb, {{a, gcomb_inv_}, {b, gx_prime}}));
  // 4. Solve for a global updater `H` that can update the final result that has been reduced by
  // the comm-reducer `f`. We will prove that
  // h(f(y_1, y_2), x, x') == f(h(y_1, x, x'), h(y_2, x, x')).
  // If so, `H` is just `h` and we're done. This covers a lot of common cases.
  const auto& prover = FindAndCheckFunc("arith.sympy.prove");
  auto dtype = reducer->result[0]->dtype;
  Var y1("y1", dtype), y2("y2", dtype);
  auto h_subst = [&](PrimExpr r_) { return Substitute(h, {{r, r_}}); };
  PrimExpr y1fy2 = reducer->operator()({y1}, {y2})[0],
           hy1fhy2 = reducer->operator()({h_subst(y1)}, {h_subst(y2)})[0],
           condition = prover(h_subst(y1fy2), hy1fhy2, "eq");
  ICHECK(is_const_int(condition, 1)) << "Cannot prove the correctness of the global "
                                        "updater `H`.";
  Memoizer::Set(cache_key, SExprPrinter::Print(h));
  return h;
}

ReduceRepairer ReduceRepairer::DeriveFromBlockExpr(const CommReducer& reducer,
                                                   PrimExpr combiner_rhs,
                                                   const ReduceBlockFrontier& frontier) {
  // 1. Examine BufferLoad usages in reduction RHS. Find `BufferLoad`s that load from incomplete
  // buffers (produced by block in `incomp_buf_info`), and replace them with temp variables `x{i}`.
  // Replace other loads with `u{i}`.
  std::unordered_map<const BufferLoadNode*, std::pair<Var, bool>> buf_load_subst;
  bool has_incomplete = false;
  PrimExpr g_expr = ReplaceBufferLoads(combiner_rhs, [&](const BufferLoadNode* buf_load) {
    auto it = buf_load_subst.find(buf_load);
    if (it != buf_load_subst.end()) {
      return it->second.first;
    }
    bool is_incomplete = frontier.HasIncompleteBuffer(buf_load->buffer);
    has_incomplete |= is_incomplete;
    std::string name = is_incomplete ? "x" : "u";
    Var x(name + std::to_string(buf_load_subst.size()), buf_load->dtype);
    buf_load_subst.emplace(buf_load, std::make_pair(x, is_incomplete));
    return x;
  });
  // If there is no incomplete buffer usage, the updater function is trivial.
  if (!has_incomplete) {
    return ReduceRepairer();
  }
  // 2. Solve the core problem: find a function `H` that can update the result of the reduction.
  // We count 2N+1 variables that the result h-expr may contain: N x-vars, N x'-vars, and 1 r-var.
  Map<Var, Var> x2xp;
  BufAndVxT buf_and_vx;
  for (auto& [buf_load, x_pair] : buf_load_subst) {
    auto [x, is_incomplete] = x_pair;
    if (is_incomplete) {
      Var x_ = x.copy_with_name(x->name_hint + "_");
      x2xp.Set(x, x_);
      buf_and_vx.emplace_back(GetRef<BufferLoad>(buf_load), x, x_);
    }
  }
  Var r("r", g_expr->dtype);
  PrimExpr h_expr = HExprSolver(reducer, g_expr, x2xp, r);
  return ReduceRepairer(h_expr, r, buf_and_vx);
}

Block ReduceRepairer::RewriteBlockExpr(Block block, const Map<Var, PrimExpr>& iv_transform,
                                       BufLoadRewriter buf_load_rewriter, PrimExpr load_condition,
                                       arith::Analyzer& analyzer, bool apply_to_lhs) const {
  if (!h_expr_.defined()) {
    return block;
  }
  auto red_match = MatchSelfReduction(NullOpt, block, std::nullopt);
  auto new_store = red_match.update;
  // Populate a substitution map from x-vars, x'-vars, and r-var to their expressions.
  Map<Var, PrimExpr> vx_to_expr;
  for (auto& [buf_load, x, x_] : buf_and_vx_) {
    auto new_buf_load = Downcast<BufferLoad>(Substitute(buf_load, iv_transform));
    auto [prev_buf_load, curr_buf_load] = buf_load_rewriter(new_buf_load);
    vx_to_expr.Set(x, prev_buf_load);
    vx_to_expr.Set(x_, curr_buf_load);
  }
  PrimExpr our_expr = red_match.rhs, their_expr = red_match.lhs;
  if (apply_to_lhs) {
    std::swap(our_expr, their_expr);
  }
  vx_to_expr.Set(vr_, our_expr);
  // Produce the new expr (could be LHS or RHS) from `vx_to_expr`.
  auto new_expr = Substitute(h_expr_, vx_to_expr);
  if (load_condition.defined()) {
    new_expr = Select(load_condition, new_expr, our_expr);
  }
  new_expr = analyzer.Simplify(new_expr);
  // Apply the reducer to the other side of the reduction to produce the full expression on the
  // right of the assignment.
  PrimExpr lhs_expr = their_expr, rhs_expr = new_expr;
  if (apply_to_lhs) {
    std::swap(lhs_expr, rhs_expr);
  }
  PrimExpr full_expr = red_match.reducer->operator()({lhs_expr}, {rhs_expr})[0];
  // Produce the full BufferStore, and apply to the Block.
  new_store.CopyOnWrite()->value = full_expr;
  {
    auto block_ptr = block.CopyOnWrite();
    block_ptr->body = new_store;
    std::tie(block_ptr->reads, block_ptr->writes) = GetBlockReadWriteRegion(block);
  }
  return block;
}

BlockRealize WithAnnotation(BlockRealize block_realize, const String& attr_name,
                            const ObjectRef& value) {
  auto br_ptr = block_realize.CopyOnWrite();
  br_ptr->block = WithAnnotation(br_ptr->block.get(), attr_name, value);
  return block_realize;
}

RFactorInternalProducts RFactorInternal(ScheduleState self, const StmtSRef& scope_root_sref,
                                        const StmtSRef& block_sref, const StmtSRef& loop_sref,
                                        int factor_axis, bool add_annotations) {
  // Get some basic information about the block and its loops.
  auto loops = GetLoops(block_sref);
  std::vector<For> inner_loops = [&] {
    auto it = std::find(loops.begin(), loops.end(), loop_sref);
    ICHECK(it != loops.end());
    return support::map(
               it + 1, loops.end(),  // Don't include loop_sref itself
               [](const StmtSRef& loop_sref) { return GetRef<For>(TVM_SREF_TO_FOR(loop_sref)); })
        .to_vector();
  }();
  auto loop = GetRef<For>(TVM_SREF_TO_FOR(loop_sref));
  auto block_realize = GetBlockRealize(self, block_sref);
  auto block = block_realize->block;
  std::unordered_set<const VarNode*> reduce_loop_vars;
  GetVarsTouchedByBlockIters(block_realize, nullptr, &reduce_loop_vars);

  // Match this block against TVM-known reduction patterns `out[i...] = f(out[i...], in[i..., j])`,
  // and extract `reducer := f`, `red_lhs := out[i...]`, `red_rhs := in[i..., j]`.
  auto red_match = tir::MatchSelfReduction(self, block, std::nullopt);
  auto buf_store = red_match.update;
  // Check and normalize `factor_axis`. If it's negative, warp it back to positive.
  factor_axis =
      FactorAxisOutOfRangeError::CheckAndUpdate(self->mod, buf_store->buffer, factor_axis);
  // Create the intermediate buffer (a.k.a. rfactor buffer), which has an additional dimension that
  // specified by `factor_axis` and `rf_loop`.
  Array<Buffer> rf_buffers = CreateExpandedBuffers({buf_store}, factor_axis, loop->extent);
  ICHECK(rf_buffers.size() == 1);
  // Create the rfactor block, then wrap it with loops.
  auto loopvars2loop = GetLoopVar2LoopMap(LoopSRefs2Loops(loops));
  RFactorBlockCreator rf_block_creator(block_realize, loop, {buf_store}, red_match.reducer,
                                       rf_buffers, loopvars2loop, factor_axis, {red_match.rhs});
  auto rf_br = rf_block_creator.CreateBlock();
  // Add the annotation to the rfactor block: (False, writeback_buffer, factor_axis)
  if (add_annotations) {
    rf_br = WithAnnotation(
        rf_br, attr::tir_rfactor_data,
        Array<ObjectRef>{Bool(false), red_match.update->buffer, Integer(factor_axis)});
  }
  Stmt rf_body = CreateLoopOutsideRfactorBlock(rf_br, inner_loops);

  // Create the write-back block.
  WriteBackBlockCreator wb_block_creator(block_realize, loop, {buf_store}, red_match.reducer,
                                         rf_buffers, std::move(rf_block_creator.additional_iter_),
                                         {red_match.lhs},
                                         std::move(rf_block_creator.rf_buf_access_indices_));
  auto wb_br = wb_block_creator.CreateBlock();
  // Add the annotation to the writeback block: (True, rfactor_buffer, factor_axis)
  if (add_annotations) {
    wb_br = WithAnnotation(wb_br, attr::tir_rfactor_data,
                           Array<ObjectRef>{Bool(true), rf_buffers[0], Integer(factor_axis)});
  }

  // Use `BlockReplacer` to edit the entire scope block, including inserting the new rf and wb
  // blocks.
  Block old_scope_root_block = GetRef<Block>(TVM_SREF_TO_BLOCK(scope_root_sref));
  auto new_scope_root_block =
      BlockReplacer::Replace(old_scope_root_block, rf_body, inner_loops[0], wb_br, block_realize,
                             loop, reduce_loop_vars, loopvars2loop, rf_buffers);
  self->Replace(scope_root_sref, new_scope_root_block,
                {{old_scope_root_block, new_scope_root_block}, {block, wb_br->block}});
  auto new_rf_sref = self->stmt2ref.at(rf_br->block.get());
  return RFactorInternalProducts{.rf_buffer = rf_buffers[0],
                                 .rf_block_sref = new_rf_sref,
                                 .wb_block_creator = wb_block_creator};
}

}  // namespace

/******** InstructionKind Registration ********/

struct DecomposeReductionTraits : public UnpackedInstTraits<DecomposeReductionTraits> {
  static constexpr const char* kName = "DecomposeReduction";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, LoopRV loop_rv) {
    return sch->DecomposeReduction(block_rv, loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, String loop_rv) {
    PythonAPICall py("decompose_reduction");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct RFactorTraits : public UnpackedInstTraits<RFactorTraits> {
  static constexpr const char* kName = "RFactor";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, Integer factor_axis,
                                         Bool merge_loops) {
    return sch->RFactor(loop_rv, factor_axis->value, merge_loops->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Integer factor_axis,
                                 Bool merge_loops) {
    PythonAPICall py("rfactor");
    py.Input("loop", loop_rv);
    py.Input("factor_axis", factor_axis->value);
    py.Input("merge_loops", merge_loops->value);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

#define CREATE_ROLLING_UPDATE_TRAITS(NAME_UPPER, NAME_LOWER)                               \
  struct NAME_UPPER##Traits : public UnpackedInstTraits<NAME_UPPER##Traits> {              \
    static constexpr const char* kName = #NAME_UPPER;                                      \
    static constexpr bool kIsPure = false;                                                 \
                                                                                           \
   private:                                                                                \
    static constexpr size_t kNumInputs = 2;                                                \
    static constexpr size_t kNumAttrs = 1;                                                 \
    static constexpr size_t kNumDecisions = 0;                                             \
                                                                                           \
    static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, LoopRV loop_rv, \
                                           Integer factor_axis) {                          \
      return sch->NAME_UPPER(block_rv, loop_rv, factor_axis->value);                       \
    }                                                                                      \
                                                                                           \
    static String UnpackedAsPython(Array<String> outputs, String block_rv, String loop_rv, \
                                   Integer factor_axis) {                                  \
      PythonAPICall py(#NAME_LOWER);                                                       \
      py.Input("block", block_rv);                                                         \
      py.Input("loop", loop_rv);                                                           \
      py.Input("factor_axis", factor_axis);                                                \
      return py.Str();                                                                     \
    }                                                                                      \
                                                                                           \
    template <typename>                                                                    \
    friend struct ::tvm::tir::UnpackedInstTraits;                                          \
  };

CREATE_ROLLING_UPDATE_TRAITS(RollingUpdate, rolling_update)
CREATE_ROLLING_UPDATE_TRAITS(SplitKUpdate, split_k_update)

TVM_REGISTER_INST_KIND_TRAITS(RFactorTraits);
TVM_REGISTER_INST_KIND_TRAITS(DecomposeReductionTraits);
TVM_REGISTER_INST_KIND_TRAITS(RollingUpdateTraits);
TVM_REGISTER_INST_KIND_TRAITS(SplitKUpdateTraits);

/******** FFI ********/

TVM_REGISTER_GLOBAL("tir.schedule.RegisterReducer")
    .set_body_typed([](int n_buffers, PackedFunc combiner_getter, PackedFunc identity_getter) {
      ReducerRegistry::RegisterReducer(n_buffers, std::move(combiner_getter),
                                       std::move(identity_getter));
    });

}  // namespace tir
}  // namespace tvm
