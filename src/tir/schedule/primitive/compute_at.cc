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
#include "../utils.h"
#include "tvm/tir/schedule/state.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

/******** Error Classes ********/

/*!
 * \brief An error raised when not all required blocks are under the given loop.
 * \tparam is_consumer Indicates if all the required blocks are consumers or producers
 */
template <bool is_consumer>
class NotAllRequiredBlocksAreVisitedError : public ScheduleError {
 public:
  explicit NotAllRequiredBlocksAreVisitedError(IRModule mod, int num_not_visited,
                                               const Array<StmtSRef>& required)
      : mod_(mod), num_not_visited_(num_not_visited) {
    required_.reserve(required.size());
    for (const StmtSRef& block_sref : required) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
      required_.push_back(GetRef<Block>(block));
    }
  }

  String FastErrorString() const final {
    return "ScheduleError: Not all required blocks are under the loop scope";
  }

  String DetailRenderTemplate() const final {
    String relation = is_consumer ? "consumer(s)" : "producer(s)";
    std::ostringstream os;
    os << "The primitive requires all the " << relation
       << " of the given block to be present under the target loop. However, there are "
       << num_not_visited_ << " " << relation << " not satisfying the constraint. List of the "
       << relation << ":";
    for (int i = 0, n = required_.size(); i < n; ++i) {
      os << "{" << i << "}";
    }
    return os.str();
  }

  IRModule mod() const final { return mod_; }

  Array<ObjectRef> LocationsOfInterest() const final {
    return {required_.begin(), required_.end()};
  }

 private:
  IRModule mod_;
  int num_not_visited_;
  Array<Block> required_;
};

/*!
 * \brief An error raised when the given block is not in the same block scope as the given loop,
 * or the given loop is the ancestor of the given block.
 */
class NotInSameScopeError : public ScheduleError {
 public:
  static void CheckAndBindLoopDomain(const ScheduleState& self, const StmtSRef& block_sref,
                                     const StmtSRef& loop_sref, const StmtSRef& scope_root_sref,
                                     arith::Analyzer* analyzer) {
    for (const StmtSRefNode* p = loop_sref.get();; p = p->parent) {
      if (const ForNode* loop = p->StmtAs<ForNode>()) {
        analyzer->Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      } else if (p != scope_root_sref.get()) {
        throw NotInSameScopeError(self->mod, block_sref, loop_sref);
      } else {
        break;
      }
    }
    for (const StmtSRefNode* p = block_sref->parent; p != scope_root_sref.get(); p = p->parent) {
      if (p == loop_sref.get()) {
        throw NotInSameScopeError(self->mod, block_sref, loop_sref);
      }
    }
  }

  String FastErrorString() const final {
    return "ScheduleError: Expected the block and loop to be under the same block scope, and loop "
           "not to be the ancestor of block";
  }
  String DetailRenderTemplate() const final {
    return "ScheduleError: Expected the block {0} and loop {1} to be under the same block scope, "
           "and loop not to be the ancestor of block";
  }
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_, loop_}; }

 private:
  explicit NotInSameScopeError(IRModule mod, const StmtSRef& block_sref, const StmtSRef& loop_sref)
      : mod_(mod),
        block_(GetRef<Block>(block_sref->StmtAs<BlockNode>())),
        loop_(GetRef<For>(loop_sref->StmtAs<ForNode>())) {}

  IRModule mod_;
  Block block_;
  For loop_;
};

class ProducerNotCompleteError : public ScheduleError {
 public:
  String FastErrorString() const final {
    return "ScheduleError: Producer of the block is a reduction block that "
           "does not produce "
           "complete elements";
  }
  String DetailRenderTemplate() const final {
    return "ScheduleError: Producer {0} is a reduction block that does not "
           "produce complete elements";
  }
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {producer_blk}; }

  ProducerNotCompleteError(IRModule mod, const Block& producer_blk)
      : mod_(mod), producer_blk(producer_blk) {}

  IRModule mod_;
  Block producer_blk;
};

/******** Helper Functions/Classes ********/

/*!
 * \brief Find a point where the block can be inserted under the loop
 * \tparam require_all_producers_visited Requires all producer blocks to be present under the loop
 * \tparam require_all_consumers_visited Requires all consumer blocks to be present under the loop
 * \param self The schedule state
 * \param subtrees The subtrees under the loop, among which the insertion points are sought
 * \param producer_srefs The producer blocks
 * \param consumer_srefs The consumer blocks
 * \param block2realize A cache that maps a block to its realize
 * \param index The block index of the loop body subtree blocks:
 * - `index = -1` means inserted into the last possible insertion point;
 * - `index = -2` means inserted into the first possible insertion point;
 * - Otherwise, `index` is a nonnegative number that indicates the insertion point
 * \return The possible position the new block can be inserted into, and the
 * producer-consumer-relationship is still satisfied.
 * \throws ScheduleError if there is no such insertion point found
 */
template <bool require_all_producers_visited, bool require_all_consumers_visited>
int FindInsertionPoint(const ScheduleState& self, const Stmt& root, const Stmt& loop_body,
                       const Array<StmtSRef>& producer_srefs, const Array<StmtSRef>& consumer_srefs,
                       std::unordered_map<const BlockNode*, const BlockRealizeNode*>* block2realize,
                       int index) {
  ProducerConsumerSplit split = ProducerConsumerSplit::Find(self, root, loop_body, producer_srefs,
                                                            consumer_srefs, block2realize);
  // Step 1. Check if all the producers are visited in the subtrees, if required to
  if (require_all_producers_visited) {
    int num_producers = producer_srefs.size();
    if (split.n_producers_visited < num_producers) {
      throw NotAllRequiredBlocksAreVisitedError<false>(
          self->mod, num_producers - split.n_producers_visited, producer_srefs);
    }
  }
  // Step 2. Check if all the consumers are visited in the subtrees, if required to
  if (require_all_consumers_visited) {
    int num_consumers = consumer_srefs.size();
    if (split.n_consumers_visited < num_consumers) {
      throw NotAllRequiredBlocksAreVisitedError<true>(
          self->mod, num_consumers - split.n_consumers_visited, consumer_srefs);
    }
  }
  // Step 3. Check if there is at least one index of the position can be inserted into
  // The valid indices are: (last_producer_position, first_consumer_position]
  ICHECK(split.last_producer_position < split.first_consumer_position);
  // Step 4. Return the possible insertion point according to index
  int insert_position;
  if (index == -1) {
    insert_position = split.first_consumer_position;
  } else if (index == -2) {
    insert_position = split.last_producer_position + 1;
  } else if (index >= 0 && index >= split.last_producer_position + 1 &&
             index <= split.first_consumer_position) {
    insert_position = index;
  } else {
    LOG(FATAL) << "Valid index:(-1, -2, [" << split.last_producer_position + 1 << ", "
               << split.first_consumer_position << "]), "
               << "current index=" << index;
    throw;
  }
  return insert_position;
}

/*!
 * \brief Represent the iteration domain to fully cover the required region of Intersect(dom, bound)
 * The bound region may not get directly intersected with dom region, instead we try to generate
 * extra predicates for non-trivial bound. The domain info class can also join with each other.
 */
struct BlockVarDomainInfo {
  arith::IntSet dom;
  arith::IntSet bound;

  BlockVarDomainInfo() = default;

  BlockVarDomainInfo(arith::IntSet dom, arith::IntSet bound)
      : dom(std::move(dom)), bound(std::move(bound)) {}

  BlockVarDomainInfo(bool is_nothing) {
    if (is_nothing) {
      dom = arith::IntSet::Nothing();
      bound = arith::IntSet::Nothing();
    } else {
      dom = arith::IntSet::Everything();
      bound = arith::IntSet::Everything();
    }
  }

  /*! \brief Relaxed union or intersect operation */
  void Join(const BlockVarDomainInfo& other, bool is_compute_at) {
    if (is_compute_at) {
      // just relax (d0 ^ b0) v (d1 ^ b1) to (d0 v d1) ^ (b0 v b1)
      dom = arith::Union({dom, other.dom});
      bound = arith::Union({bound, other.bound});
    } else {
      dom = arith::Intersect({dom, other.dom});
      bound = arith::Intersect({bound, other.bound});
    }
  }

  /*! \brief Simplify domain info */
  void Simplify(arith::Analyzer* analyzer) {
    auto to_simplified = [analyzer](const arith::IntSet& set) {
      PrimExpr min = set.HasLowerBound() ? analyzer->Simplify(set.min()) : set.min();
      PrimExpr max = set.HasUpperBound() ? analyzer->Simplify(set.max()) : set.max();
      return arith::IntSet::Interval(min, max);
    };
    // if no dom specified, try use bound as dom
    if (dom.IsNothing()) {
      if (bound.HasLowerBound() && bound.HasUpperBound()) {
        bound = to_simplified(bound);
        std::swap(dom, bound);
      }
      return;
    }
    // simplify intset
    dom = to_simplified(dom);
    bound = to_simplified(bound);
    // if can proof the dom is within bound, remove bound
    auto intersect = to_simplified(arith::Intersect({dom, bound}));
    if (analyzer->CanProveEqual(dom.min(), intersect.min()) &&
        analyzer->CanProveEqual(dom.max(), intersect.max())) {
      bound = arith::IntSet::Nothing();
    } else if (analyzer->CanProveEqual(bound.min(), intersect.min()) &&
               analyzer->CanProveEqual(bound.max(), intersect.max())) {
      dom = bound;
      bound = arith::IntSet::Nothing();
    } else if (is_const_int(intersect.min()) && is_const_int(intersect.max())) {
      // if the bound induce constant iter range, merge bound to loop domain
      dom = intersect;
      bound = arith::IntSet::Nothing();
    }
  }
};

class ScopeReconstructor : private StmtMutator {
 public:
  static ScopeReconstructor CreateWithRemovalPlan(const ScheduleState& self, Block scope_root,
                                                  StmtSRef block_sref, Variant<Block, For> target) {
    Stmt rm_src_stmt, rm_tgt_stmt;
    LeafBlockRemovalPlan(self, block_sref, &rm_src_stmt, &rm_tgt_stmt);
    auto block = GetRef<Block>(TVM_SREF_TO_BLOCK(block_sref));
    return ScopeReconstructor(std::move(scope_root), std::move(block), std::move(target),
                              std::move(rm_src_stmt), std::move(rm_tgt_stmt));
  }

  /*!
   * \brief Create the loop nest on top of the block, induced by the given block var's domain
   * \param insert_position The position among the subtrees where the block and its induced loop
   * nest is inserted
   * \param iter_doms The domain of each block var
   * \param analyzer The arithmetic analyzer
   * \param preserve_unit_loops Whether to generate unit loops where the loop extent is 1
   */
  void MakeNewLoop(int insert_position, std::vector<BlockVarDomainInfo> iter_doms,
                   arith::Analyzer* analyzer, bool preserve_unit_loops) {
    int n_iters = iter_doms.size();
    Array<Var> loop_vars;
    Array<PrimExpr> loop_extents;
    Array<PrimExpr> iter_values;
    loop_vars.reserve(n_iters);
    loop_extents.reserve(n_iters);
    iter_values.reserve(n_iters);
    PrimExpr predicate = const_true();
    for (int i = 0; i < n_iters; ++i) {
      Range iter_dom = iter_doms[i].dom.CoverRange(block_->iter_vars[i]->dom);
      if (preserve_unit_loops || !is_one(iter_dom->extent)) {
        int bits = std::max(iter_dom->min.dtype().bits(), iter_dom->extent.dtype().bits());
        Var var("ax" + std::to_string(loop_vars.size()), DataType::Int(bits));
        loop_vars.push_back(var);
        loop_extents.push_back(analyzer->Simplify(iter_dom->extent));
        iter_values.push_back(iter_dom->min + var);
        analyzer->Bind(var, Range::FromMinExtent(IntImm(var.dtype(), 0), iter_dom->extent));
      } else {
        iter_values.push_back(iter_dom->min);
      }
      const arith::IntSet& pred_bound = iter_doms[i].bound;
      if (!pred_bound.IsNothing()) {
        // NOTE: Apply strong analyzer proofs to get rid of symbolic bound
        if (pred_bound.HasLowerBound()) {
          PrimExpr lower_bound = iter_values[i] >= pred_bound.min();
          if (!analyzer->CanProve(lower_bound, arith::ProofStrength::kSymbolicBound)) {
            predicate = predicate && lower_bound;
          }
        }
        if (pred_bound.HasUpperBound()) {
          PrimExpr upper_bound = iter_values[i] < pred_bound.max() + 1;
          if (!analyzer->CanProve(upper_bound, arith::ProofStrength::kSymbolicBound)) {
            predicate = predicate && upper_bound;
          }
        }
      }
    }
    this->new_block_realize_ =
        BlockRealize(std::move(iter_values), analyzer->Simplify(predicate), std::move(block_));
    Stmt new_subtree = this->new_block_realize_;
    for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
      const Var& loop_var = loop_vars[i];
      const PrimExpr& loop_extent = loop_extents[i];
      new_subtree = For(/*loop_var=*/loop_var,
                        /*min=*/Integer(0),
                        /*extent=*/loop_extent,
                        /*ForKind=*/ForKind::kSerial,
                        /*body=*/std::move(new_subtree));
    }
    auto GetBody = [](auto&& block) { return block->body; };
    auto target_body = tir::VisitSRefStmtVariant(target_, GetBody, GetBody);
    Array<Stmt> subtrees = AsArray(target_body);
    subtrees.insert(subtrees.begin() + insert_position, std::move(new_subtree));
    auto new_body = SeqStmt(std::move(subtrees));
    this->new_target_ = tir::VisitSRefStmtVariant(
        target_,
        [new_body](const Block& block) -> Stmt {
          auto new_stmt = make_object<BlockNode>(*block.get());
          new_stmt->body = std::move(new_body);
          return Stmt(std::move(new_stmt));
        },
        [new_body](const For& loop) -> Stmt {
          auto new_stmt = make_object<ForNode>(*loop.get());
          new_stmt->body = std::move(new_body);
          return Stmt(std::move(new_stmt));
        });
  }

  using StmtMutator::operator();

 private:
  explicit ScopeReconstructor(Block scope_root, Block block, Variant<Block, For> target,
                              Stmt rm_src_stmt, Stmt rm_tgt_stmt)
      : scope_root_(std::move(scope_root)),
        block_(std::move(block)),
        target_(std::move(target)),
        rm_src_stmt_(std::move(rm_src_stmt)),
        rm_tgt_stmt_(std::move(rm_tgt_stmt)) {
    ICHECK(!rm_src_stmt_.same_as(target_)) << "The removal target (where the block is erased from) "
                                              "cannot be the same as the target stmt where "
                                              "the block is moved to";
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    auto new_stmt = stmt;
    if (new_stmt.same_as(target_)) {
      // 1. Replace the compute-at target with the new target, which will create the new version of
      // `block_`.
      new_stmt = new_target_;
    } else if (new_stmt.same_as(rm_src_stmt_)) {
      // 2. Try and apply the removal plan to remove `block_` from its old position. This case
      // cannot overlap with 1 because we checked that `target_` is never the same as
      // `rm_src_stmt_`.
      new_stmt = rm_tgt_stmt_;
    }
    return StmtMutator::VisitStmt(new_stmt);
  }

  /*! \brief The root block of the block scope. */
  Block scope_root_;
  /*! \brief The block to operate on. Equivalent to the block argument of compute-at. */
  Block block_;
  /*! \brief The stmt to put `block_` under. Equivalent to the loop argument of compute-at. */
  Variant<Block, For> target_;
  /*! \brief A removal plan. By replacing `rm_src_stmt_` with `rm_tgt_stmt_`, we can remove the
   * given block from its old position. */
  Stmt rm_src_stmt_, rm_tgt_stmt_;
  /*! \brief The new loop to replace the original loop */
  Stmt new_target_{nullptr};

 public:
  /*! \brief The new block realize to the moved block */
  BlockRealize new_block_realize_{nullptr};
};

using BufferRegionsT = std::unordered_map<const BufferNode*, std::vector<NDIntSet>>;
/*!
 * \brief Calculate a list of accessed buffer regions under a path of loops
 * \tparam relax_storage_scope Whether to relax beyond the path according to the storage and
 * execution scope
 * \param binding The block binding, used to unbind the buffer regions
 * \param block_sref The block sref to be analyzed.
 * \param buffer_regions The buffer regions to be calculated
 * \param relax_path_high_exclusive The highest point in the loop path, exclusive
 * \param check_reduce_completeness If false, skip domain completeness check for reduction
 * iter-vars. Has no effect when `is_compute_at` is True.
 * \param relaxed Where the calculation result is stored
 */
template <bool is_compute_at>
void RelaxBufferRegions(const ScheduleState& self, const StmtSRef& block_sref,
                        const BlockRealize& block_realize,
                        const StmtSRef& relax_path_high_exclusive, bool check_reduce_completeness,
                        arith::Analyzer& analyzer, BufferRegionsT& relaxed) {
  Block block = block_realize->block;
  Map<Var, PrimExpr> binding = GetBindings(block_realize);
  auto relax_path_low_inclusive = GetRef<StmtSRef>(block_sref->parent);
  runtime::StorageScope global_scope{runtime::StorageRank::kGlobal, ""};
  // We cache the variable domains
  runtime::StorageRank previous_rank = runtime::StorageRank::kGlobal;
  Optional<Map<Var, arith::IntSet>> var_dom = NullOpt;
  // In reverse_compute_at mode, prepare to check reduction completeness
  auto IsReductionComplete = [&](const Map<Var, arith::IntSet>& var_dom) {
    for (size_t i = 0; i < block->iter_vars.size(); ++i) {
      auto iter_var = block->iter_vars[i];
      if (iter_var->iter_type != kCommReduce) {
        continue;
      }
      // IntSet is inclusive on both sides [a, b] while Range is [a, b)
      arith::IntSet iv_relaxed_dom = arith::EvalSet(block_realize->iter_values[i], var_dom);
      Range iv_dom = iter_var->dom;
      if (!analyzer.CanProve(iv_relaxed_dom.min() == iv_dom->min) ||
          !analyzer.CanProve(iv_relaxed_dom.max() + 1 == iv_dom->extent)) {
        return false;
      }
    }
    return true;
  };
  // Enumerate every buffer region
  for (const BufferRegion& buffer_region : is_compute_at ? block->reads : block->writes) {
    const Buffer& buffer = buffer_region->buffer;
    const Array<Range>& region = buffer_region->region;
    // Skip the buffer regions we are not interested in
    auto it = relaxed.find(buffer.get());
    if (it == relaxed.end()) {
      continue;
    }
    std::vector<NDIntSet>& relaxed_regions = it->second;
    // Check and update the cached `var_dom`
    runtime::StorageScope scope =
        is_compute_at ? runtime::StorageScope::Create(buffer.scope()) : global_scope;
    runtime::StorageRank rank = scope.rank;
    if (rank != previous_rank || !var_dom.defined()) {
      previous_rank = rank;
      var_dom = arith::AsIntSet(LoopDomainOfSRefTreePath(
          /*low_inclusive=*/relax_path_low_inclusive,
          /*high_exclusive=*/relax_path_high_exclusive,
          /*extra_relax_scope=*/scope));
    }
    // In reverse_compute_at mode, check that elements in the produced region are fully produced.
    if (!is_compute_at && check_reduce_completeness && !IsReductionComplete(var_dom.value())) {
      throw ProducerNotCompleteError(self->mod, block);
    }
    // Relax the region
    Array<arith::IntSet> relaxed_region =
        arith::EvalSet(Substitute(region, binding), var_dom.value());
    relaxed_regions.push_back({relaxed_region.begin(), relaxed_region.end()});
  }
}

/*!
 * \brief Calculate the iteration domain of an integer set (`coverer`) to fully cover another
 * (`coveree`)
 * \param coverer The integer set to cover the domain
 * \param coveree The domain to be covered
 * \param dim_max The maximum index bound by the buffer shape
 * \param analyzer The arithmetic analyzer
 * \return (var, range_info): the variable whose range is constrained by the coverer-coveree
 * relation, and the range info of the variable. If the coverer-coveree relation holds
 * unconditionally, return std::nullopt.
 */
template <bool is_compute_at>
std::optional<std::pair<Var, BlockVarDomainInfo>> SolveBlockVarDomain(const arith::IntSet& coverer,
                                                                      const arith::IntSet& coveree,
                                                                      PrimExpr dim_max,
                                                                      arith::Analyzer* analyzer) {
  PrimExpr coverer_min = analyzer->Simplify(coverer.min());
  PrimExpr coverer_max = analyzer->Simplify(coverer.max());
  PrimExpr coveree_min = analyzer->Simplify(coveree.min());
  PrimExpr coveree_max = analyzer->Simplify(coveree.max());
  arith::IntSet var_dom, var_bound;
  Optional<Var> var;
  arith::PVar<Var> p_v;
  arith::PVar<PrimExpr> p_e;
  if (is_const_int(coverer_min) && is_const_int(coverer_max)) {
    if (is_compute_at) {
      ICHECK(analyzer->CanProve(coverer_min <= coveree_min) &&
             analyzer->CanProve(coverer_max >= coveree_max));
    } else {
      ICHECK(analyzer->CanProve(coveree_min <= coverer_min) &&
             analyzer->CanProve(coveree_max >= coverer_max));
    }
    return std::nullopt;
  } else if ((p_v * p_e).Match(coverer_min) || (p_e * p_v).Match(coverer_min)) {
    PrimExpr e = p_e.Eval();
    var = p_v.Eval();
    var_dom = arith::IntSet::Interval(floordiv(coveree_min, e), floordiv(coveree_max, e));
    var_bound = arith::IntSet::Interval(0, floordiv(dim_max, e));
  } else if (analyzer->CanProveEqual(coverer_min, coverer_max)) {
    if (p_v.Match(coverer_min)) {
      var = p_v.Eval();
      var_dom = arith::IntSet::Interval(coveree_min, coveree_max);
      var_bound = arith::IntSet::Interval(0, dim_max);
    } else {
      arith::PVar<PrimExpr> p_f1, p_f2;
      if ((floordiv(p_f1, p_f2).Match(coverer_min))) {
        PrimExpr var_expr = p_f1.Eval();
        PrimExpr fac = p_f2.Eval();
        if (analyzer->CanProveGreaterEqual(fac, 1)) {
          if (var_expr->IsInstance<VarNode>()) {
            // a <= (x // factor) <= b, fac > 0 ==> (a * fac) <= x <= (b * fac + fac - 1)
            var = Downcast<Var>(var_expr);
            var_dom = arith::IntSet::Interval(coveree_min * fac,
                                              analyzer->Simplify(coveree_max * fac + fac - 1));
            var_bound = arith::IntSet::Interval(0, analyzer->Simplify(dim_max * fac + fac - 1));
          } else {
            auto new_coverer = arith::IntSet::SinglePoint(p_f1.Eval());
            auto new_coveree = arith::IntSet::Interval(
                coveree_min * fac, analyzer->Simplify(coveree_max * fac + fac - 1));
            return SolveBlockVarDomain<is_compute_at>(new_coverer, new_coveree, dim_max, analyzer);
          }
        }
      } else if ((floormod(p_f1, p_f2).Match(coverer_min))) {
        PrimExpr var_expr = p_f1.Eval();
        if (var_expr->IsInstance<VarNode>()) {
          // generally domain of (x % fac) enforce no constraints to domain of x
          Var var_mod = Downcast<Var>(var_expr);
          return {{var_mod, BlockVarDomainInfo(is_compute_at)}};
        } else {
          PrimExpr mod_1 = p_f1.Eval();
          PrimExpr mod_2 = p_f2.Eval();
          if (analyzer->CanProveGreaterEqual(mod_1, 1) &&
              analyzer->CanProveGreaterEqual(mod_2, 1)) {
            auto new_coverer = arith::IntSet::SinglePoint(p_f1.Eval());
            if (analyzer->CanProveGreaterEqual(coveree_min, 0)) {
              auto new_coveree =
                  arith::IntSet::Interval(coveree_min, arith::SymbolicLimits::pos_inf_);
              return SolveBlockVarDomain<is_compute_at>(new_coverer, new_coveree, dim_max,
                                                        analyzer);
            }
          }
        }
      }
    }
  }
  ICHECK(var.defined()) << "ValueError: BufferRegion pattern match failed: " << coverer_min << ", "
                        << coverer_max << ", " << coveree_min << ", " << coveree_max;
  return {{var.value(), BlockVarDomainInfo{var_dom, var_bound}}};
}

/*!
 * \brief Calculate the iteration domain info to fully cover a region `depd_region`.
 * This function implements dimension-wise method: the region relation on each buffer dimension is
 * independently estimated.
 * \param buffer The accessed buffer
 * \param self_region The region to cover `depd_region` (the "coverer")
 * \param depd_region The region to be covered (the "coveree")
 * \param analyzer The arithmetic analyzer
 * \param iter_doms The result iteration domains to be updated
 */
template <bool is_compute_at>
void UpdateBlockVarDomainDimwise(
    const BufferNode* buffer, const NDIntSet& self_region, const NDIntSet& depd_region,
    arith::Analyzer* analyzer, std::unordered_map<const VarNode*, BlockVarDomainInfo>* iter_doms) {
  size_t ndim = buffer->shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    arith::IntSet self = self_region[i];
    arith::IntSet depd = depd_region[i];
    PrimExpr dim_max = max(buffer->shape[i] - 1, 0);
    auto result = SolveBlockVarDomain<is_compute_at>(self, depd, dim_max, analyzer);
    if (!result.has_value()) {
      continue;
    }
    auto [var, dom_info] = result.value();
    if (auto it = iter_doms->find(var.get()); it != iter_doms->end()) {
      it->second.Join(dom_info, is_compute_at);
    } else {
      ICHECK(analyzer->CanProveEqual(self.min(), depd.min()));
      ICHECK(analyzer->CanProveEqual(self.max(), depd.max()));
    }
  }
}

/*! \brief Helper function to implement intset version of `InverseAffineIterMap`. */
Map<Var, arith::IntSet> InverseAffineIterMap(const Array<arith::IterSumExpr>& iter_map,
                                             const NDIntSet& outputs, arith::Analyzer* analyzer) {
  Array<PrimExpr> min_point, max_point;
  min_point.reserve(outputs.size());
  max_point.reserve(outputs.size());
  for (const auto& intset : outputs) {
    ICHECK(intset.HasLowerBound() && intset.HasUpperBound());
    min_point.push_back(intset.min());
    max_point.push_back(intset.max());
  }
  auto rev_min = InverseAffineIterMap(iter_map, min_point);
  auto rev_max = InverseAffineIterMap(iter_map, max_point);
  Map<Var, arith::IntSet> dom_map;
  for (const auto& kv : rev_min) {
    const Var& var = kv.first;
    auto it = rev_max.find(var);
    ICHECK(it != rev_max.end());  // InverseAffineIterMap's result vars are assumed stable
    const PrimExpr& rev_min_point = kv.second;
    const PrimExpr& rev_max_point = (*it).second;
    dom_map.Set(var,
                arith::IntSet::Interval(analyzer->Simplify(min(rev_min_point, rev_max_point)),
                                        analyzer->Simplify(max(rev_min_point, rev_max_point))));
  }
  return dom_map;
}

/*!
 * \brief Calculate the iteration domain info to fully cover a region `depd_region`.
 * This function implements affine analysis. It requires bijective mapping of block var to
 * `depd_region` points.
 * \param buffer The accessed buffer
 * \param iter_vars The list of block vars to cover the required region
 * \param self_region The region to cover `depd_region` (the "coverer")
 * \param depd_region The region to be covered (the "coveree")
 * \param analyzer The arithmetic analyzer
 * \param iter_doms The result iteration domains to be updated
 * \returns bool. Denotes whether update success
 */
template <bool is_compute_at>
bool UpdateBlockVarDomainAffine(const BufferNode* buffer, const Array<IterVar>& iter_vars,
                                const NDIntSet& self_region, const NDIntSet& depd_region,
                                arith::Analyzer* analyzer,
                                std::unordered_map<const VarNode*, BlockVarDomainInfo>* iter_doms) {
  // we only support single point depd_region now, which could cover most cases
  for (const auto& intset : self_region) {
    if (!intset.CanProveSinglePoint(analyzer)) return false;
  }
  // calculate forward mapping (block vars -> depd_region point)
  Map<Var, Range> dom_map;
  for (const IterVar& iter_var : iter_vars) {
    dom_map.Set(iter_var->var, iter_var->dom);
  }
  size_t ndim = buffer->shape.size();
  Array<PrimExpr> provide_indices;
  provide_indices.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    provide_indices.push_back(self_region[i].min());
  }
  auto res = arith::DetectIterMap(provide_indices, dom_map, const_true(),
                                  arith::IterMapLevel::Bijective, analyzer, false);
  if (res->indices.empty()) {
    return false;
  }
  // calculate backward mapping (depd_region point -> block vars)
  NDIntSet depd_bound;
  for (size_t i = 0; i < ndim; ++i) {
    depd_bound.push_back(
        arith::IntSet::Interval(make_zero(buffer->shape[i]->dtype), max(buffer->shape[i] - 1, 0)));
  }
  Map<Var, arith::IntSet> var_dom = InverseAffineIterMap(res->indices, depd_region, analyzer);
  Map<Var, arith::IntSet> var_bound = InverseAffineIterMap(res->indices, depd_bound, analyzer);
  for (const auto& kv : var_dom) {
    const Var& var = kv.first;
    auto it = var_bound.find(var);
    ICHECK(it != var_bound.end());  // InverseAffineIterMap's result vars are assumed stable
    BlockVarDomainInfo &lhs = (*iter_doms)[var.get()], rhs{kv.second, (*it).second};
    lhs.Join(rhs, is_compute_at);
  }
  return true;
}

/*!
 * \brief Calculate the domain of block vars to cover the dependent region
 * \param iter_vars The list of block vars to cover the dependent region
 * \param self_regions The region covered in one iteration of the block vars
 * \param depd_regions The region to be covered
 * \param analyzer The arithmetic analyzer
 * \return A list of iteration domain info corresponding to the given list of block vars
 */
template <bool is_compute_at>
std::vector<BlockVarDomainInfo> CalculateBlockVarDomain(const Array<IterVar>& iter_vars,
                                                        const BufferRegionsT& self_regions,
                                                        const BufferRegionsT& depd_regions,
                                                        arith::Analyzer* analyzer) {
  int n_iters = iter_vars.size();
  // Step 1. Construct the mapping from block var to their iteration domain (initialized to empty)
  std::unordered_map<const VarNode*, BlockVarDomainInfo> iter_doms;
  iter_doms.reserve(n_iters);
  for (const IterVar& iter_var : iter_vars) {
    iter_doms[iter_var->var.get()] = BlockVarDomainInfo(is_compute_at);
  }
  // Step 2. For each buffer, update the domain:
  for (const auto& [buffer, many_regions] : self_regions) {
    // Calculate `self_region` and `depd_region`
    auto it = depd_regions.find(buffer);
    if (it == depd_regions.end() || it->second.empty()) {
      continue;
    }
    NDIntSet depd_region = support::NDIntSetUnion(it->second);
    NDIntSet self_region = support::NDIntSetUnion(many_regions);
    ICHECK_EQ(self_region.size(), buffer->shape.size());
    ICHECK_EQ(depd_region.size(), buffer->shape.size());
    // Try update iter var domains with current self and depd region pair.
    if (!UpdateBlockVarDomainAffine<is_compute_at>(buffer, iter_vars, self_region, depd_region,
                                                   analyzer, &iter_doms)) {
      UpdateBlockVarDomainDimwise<is_compute_at>(buffer, self_region, depd_region, analyzer,
                                                 &iter_doms);
    }
  }
  // Union the iter var domains, put them in the same order of block vars, and return
  std::vector<BlockVarDomainInfo> result;
  result.reserve(n_iters);
  for (const IterVar& iter_var : iter_vars) {
    BlockVarDomainInfo& info = iter_doms.at(iter_var->var.get());
    if (info.bound.IsNothing()) {
      info.bound = arith::IntSet::FromRange(iter_var->dom);
    } else {
      info.bound = arith::Intersect({info.bound, arith::IntSet::FromRange(iter_var->dom)});
    }
    info.Simplify(analyzer);
    ICHECK(!info.dom.IsNothing());
    result.push_back(info);
  }
  return result;
}

/*!
 * \brief Calculate the regions covered by the given block in one single execution instance,
 * and the regions relaxed to the given loop
 * \tparam is_compute_at Indicates if the operation is compute-at or reverse-compute-at
 * \param block The given block that provides buffer regions
 * \param loop_sref The given loop under which the block is going to be moved to
 * \param block2realize Maps a block to its corresponding BlockRealize
 * \param producer_srefs The producers of the given block
 * \param consumer_srefs The consumers of the given block
 * \param check_reduce_completeness If false, skip domain completeness check for reduction
 * iter-vars. Has no effect when `is_compute_at` is True.
 * \return (self_regions, depd_regions): the regions covered by the block, and the regions covered
 * by its consumers (in compute-at) or producers (in reverse-compute-at)
 */
template <bool is_compute_at>
std::pair<BufferRegionsT, BufferRegionsT> CalculateSelfAndDepdRegions(
    const ScheduleState& self, const BlockNode* block, const StmtSRef& loop_sref,
    const std::unordered_map<const BlockNode*, const BlockRealizeNode*>& block2realize,
    const Array<StmtSRef>& producer_srefs, const Array<StmtSRef>& consumer_srefs,
    arith::Analyzer& analyzer, bool check_reduce_completeness = true) {
  // Step 1. Calculate the region covered by a single execution instance of `block`
  const Array<BufferRegion>& buffers = is_compute_at ? block->writes : block->reads;
  BufferRegionsT self_regions, depd_regions;
  self_regions.reserve(buffers.size());
  depd_regions.reserve(buffers.size());
  for (const BufferRegion& self_buffer_region : buffers) {
    const BufferNode* buffer = self_buffer_region->buffer.get();
    const Array<Range>& region = self_buffer_region->region;
    self_regions[buffer].push_back(support::NDIntSetFromRegion(region));
    depd_regions[buffer].clear();
  }
  // Step 2. Calculate the region of dependent blocks under `loop`
  for (const StmtSRef& depd_block_sref : is_compute_at ? consumer_srefs : producer_srefs) {
    const BlockNode* depd_block = TVM_SREF_TO_BLOCK(depd_block_sref);
    auto it = block2realize.find(depd_block);
    ICHECK(it != block2realize.end());
    RelaxBufferRegions<is_compute_at>(self, depd_block_sref, GetRef<BlockRealize>(it->second),
                                      loop_sref, check_reduce_completeness, analyzer, depd_regions);
  }
  return {self_regions, depd_regions};
}

/******** Main Implementation ********/

template <bool is_compute_at>
void ComputeAtOrReverseComputeAtImpl(ScheduleState self, const StmtSRef& block_sref,
                                     const StmtSRef& loop_sref, bool preserve_unit_loops, int index,
                                     bool check_only = false,
                                     bool check_reduce_completeness = true) {
  arith::Analyzer analyzer;
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  // Step 1. Bunch of checks
  // Check condition 1) : scope stage pipeline
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref,
                                          /*require_stage_pipeline=*/false);
  Block scope_root = GetRef<Block>(scope_root_sref->StmtAs<BlockNode>());
  AddShapeVarBounds(self, scope_root_sref.get(), &analyzer);
  BlockScope scope = self->GetBlockScope(scope_root_sref);
  Array<StmtSRef> producer_srefs = GetProducers(block_sref, scope);
  Array<StmtSRef> consumer_srefs = GetConsumers(block_sref, scope);
  // Check condition 2) : `block` is a complete or reduction block
  // CheckCompleteOrReductionBlock(self, block_sref, scope_root_sref);
  // Check condition 3): `block` and `loop` are under the same scope,
  // and `loop` is not the ancestor of `block`
  NotInSameScopeError::CheckAndBindLoopDomain(self, block_sref, loop_sref, scope_root_sref,
                                              &analyzer);
  // Check condition 4): `block` is not an output block
  if (is_compute_at) {
    CheckNotOutputBlock(self, block_sref, scope_root_sref);
  }
  // Step 2. Plan for the removal of `block`
  auto reconstructor =
      ScopeReconstructor::CreateWithRemovalPlan(self, scope_root, block_sref, GetRef<For>(loop));
  // Step 3. Find the insertion point under `loop`
  // Check condition 5): all the required block are under the given loop
  std::unordered_map<const BlockNode*, const BlockRealizeNode*> block2realize;
  block2realize.reserve(self->block_info.size());
  int insert_position = FindInsertionPoint<!is_compute_at, is_compute_at>(
      self, scope_root, loop->body, producer_srefs, consumer_srefs, &block2realize, index);
  // Step 4. Calculate the region covered in a single execution of `block`,
  // as well as the region covered by dependent blocks under `loop`.
  auto [self_regions, depd_regions] = CalculateSelfAndDepdRegions<is_compute_at>(
      self, block, loop_sref, std::move(block2realize), std::move(producer_srefs),
      std::move(consumer_srefs), analyzer, check_reduce_completeness);
  // Step 5. Calculate the iteration domain for each block var
  std::vector<BlockVarDomainInfo> iter_doms = CalculateBlockVarDomain<is_compute_at>(
      /*iter_vars=*/block->iter_vars, std::move(self_regions), std::move(depd_regions), &analyzer);
  // Step 6. Create the new scope according to the iteration domain
  reconstructor.MakeNewLoop(/*insert_position=*/insert_position, /*iter_doms=*/std::move(iter_doms),
                            /*analyzer=*/&analyzer, /*preserve_unit_loops=*/preserve_unit_loops);
  Block new_scope_root = Downcast<Block>(reconstructor(scope_root));

  // Step 7. Do the actual replacement
  if (check_only) {
    return;
  }
  self->Replace(scope_root_sref, new_scope_root, {{scope_root, new_scope_root}});
  // Step 8. Update the cached flags
  BlockInfo& block_info = self->block_info[block_sref];
  block_info.affine_binding = IsAffineBinding(
      /*realize=*/reconstructor.new_block_realize_,
      /*loop_var_ranges=*/LoopDomainOfSRefTreePath(GetRef<StmtSRef>(block_sref->parent)),
      /*analyzer=*/&analyzer);
}

void ComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
               bool preserve_unit_loops, int index) {
  ComputeAtOrReverseComputeAtImpl<true>(self, block_sref, loop_sref, preserve_unit_loops, index);
}

void ReverseComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                      bool preserve_unit_loops, int index) {
  ComputeAtOrReverseComputeAtImpl<false>(self, block_sref, loop_sref, preserve_unit_loops, index);
}

void UnsafeReverseComputeAt(ScheduleState self, const StmtSRef& block_sref,
                            const StmtSRef& loop_sref, bool preserve_unit_loops, int index) {
  ComputeAtOrReverseComputeAtImpl<false>(self, block_sref, loop_sref, preserve_unit_loops, index,
                                         /*check_only=*/false, /*check_reduce_completeness=*/false);
}

void ReverseComputeRoot(ScheduleState self, const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  arith::Analyzer analyzer;
  // Step 1. Check scope stage pipeline and get the scope root
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref,
                                          /*require_stage_pipeline=*/false);
  Block scope_root = GetRef<Block>(scope_root_sref->StmtAs<BlockNode>());
  BlockScope scope = self->GetBlockScope(scope_root_sref);
  // Step 2. Set up the analyzer
  AddShapeVarBounds(self, scope_root_sref.get(), &analyzer);
  for (const StmtSRefNode* p = block_sref.get(); p != scope_root_sref.get(); p = p->parent) {
    if (const ForNode* loop = p->StmtAs<ForNode>()) {
      analyzer.Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
  }
  // Step 3. Plan for the removal of `block`
  auto reconstructor =
      ScopeReconstructor::CreateWithRemovalPlan(self, scope_root, block_sref, scope_root);
  // Step 4. Find the insertion point under `loop`
  // Check condition 5): all the required block are under the given loop
  Array<StmtSRef> producer_srefs = GetProducers(block_sref, scope);
  Array<StmtSRef> consumer_srefs = GetConsumers(block_sref, scope);
  std::unordered_map<const BlockNode*, const BlockRealizeNode*> block2realize;
  block2realize.reserve(self->block_info.size());
  int insert_position =
      FindInsertionPoint<true, false>(self, scope_root, scope_root->body, producer_srefs,
                                      consumer_srefs, &block2realize, /*index=*/-1);
  // Step 5. Calculate the region covered in a single execution of `block`,
  // as well as the region covered by dependent blocks under `loop`.
  auto [self_regions, depd_regions] = CalculateSelfAndDepdRegions<false>(
      self, block, scope_root_sref, std::move(block2realize), std::move(producer_srefs),
      std::move(consumer_srefs), analyzer);
  // Step 5. Calculate the iteration domain for each block var
  std::vector<BlockVarDomainInfo> iter_doms = CalculateBlockVarDomain<false>(
      /*iter_vars=*/block->iter_vars, std::move(self_regions), std::move(depd_regions), &analyzer);
  // Step 6. Create the new scope according to the iteration domain
  reconstructor.MakeNewLoop(/*insert_position=*/insert_position, /*iter_doms=*/std::move(iter_doms),
                            /*analyzer=*/&analyzer, /*preserve_unit_loops=*/true);
  Block new_scope_root = Downcast<Block>(reconstructor(scope_root));
  // Step 7. Do the actual replacement
  self->Replace(scope_root_sref, new_scope_root, {{scope_root, new_scope_root}});
  // Step 8. Update the cached flags
  BlockInfo& block_info = self->block_info[block_sref];
  block_info.affine_binding = IsAffineBinding(
      /*realize=*/reconstructor.new_block_realize_,
      /*loop_var_ranges=*/LoopDomainOfSRefTreePath(GetRef<StmtSRef>(block_sref->parent)),
      /*analyzer=*/&analyzer);
}

bool CanComputeAt(const ScheduleState& self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                  bool preserve_unit_loops) {
  try {
    ComputeAtOrReverseComputeAtImpl<true>(self, block_sref, loop_sref, preserve_unit_loops,
                                          /*index=*/-1, /*check_only=*/true);
  } catch (const tvm::runtime::Error& e) {
    return false;
  }
  return true;
}

bool CanReverseComputeAt(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& loop_sref, bool preserve_unit_loops) {
  try {
    ComputeAtOrReverseComputeAtImpl<true>(self, block_sref, loop_sref, preserve_unit_loops,
                                          /*index=*/-1, /*check_only=*/true);
  } catch (const tvm::runtime::Error& e) {
    return false;
  }
  return true;
}

/******** InstructionKind Registration ********/

struct ComputeAtTraits : public UnpackedInstTraits<ComputeAtTraits> {
  static constexpr const char* kName = "ComputeAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, LoopRV loop_rv,
                                      Bool preserve_unit_loops, IntImm index) {
    return sch->ComputeAt(block_rv, loop_rv, preserve_unit_loops.operator bool(), index->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, String loop_rv,
                                 Bool preserve_unit_loops, IntImm index) {
    PythonAPICall py("compute_at");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.Input("preserve_unit_loops", preserve_unit_loops.operator bool());
    py.Input("index", index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReverseComputeAtTraits : public UnpackedInstTraits<ReverseComputeAtTraits> {
  static constexpr const char* kName = "ReverseComputeAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, LoopRV loop_rv,
                                      Bool preserve_unit_loops, IntImm index) {
    return sch->ReverseComputeAt(block_rv, loop_rv, preserve_unit_loops.operator bool(),
                                 index->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, String loop_rv,
                                 Bool preserve_unit_loops, IntImm index) {
    PythonAPICall py("reverse_compute_at");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.Input("preserve_unit_loops", preserve_unit_loops.operator bool());
    py.Input("index", index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ComputeAtTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReverseComputeAtTraits);

}  // namespace tir
}  // namespace tvm
