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
#include <tvm/arith/iter_affine_map.h>
#include <tvm/support/iterator.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/utils.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../../arith/pattern_match.h"
#include "../analysis.h"
#include "../transform.h"

namespace tvm::arith {

template <typename T>
class POptionalCast : public Pattern<POptionalCast<T>> {
 public:
  POptionalCast(const PVar<T>& var_pat) : var_pat_(var_pat) {}

  void InitMatch_() const { var_pat_.InitMatch_(); }

  bool Match_(const PrimExpr& expr) const {
    if (var_pat_.Match(expr)) {
      return true;
    }
    if (auto cast = expr.as<tir::CastNode>()) {
      return var_pat_.Match(cast->value);
    }
    return false;
  }

  T Eval() const { return var_pat_.Eval(); }

 private:
  const PVar<T>& var_pat_;
};

template <>
class PEqualChecker<tir::BufferLoad> {
 public:
  bool operator()(const tir::BufferLoad& lhs, const tir::BufferLoad& rhs) const {
    if (lhs.same_as(rhs)) return true;
    return tir::ExprDeepEqual()(lhs, rhs);
  }
};

}  // namespace tvm::arith

namespace tvm::tir {

namespace {

/*!
 * \brief A pair of a region and the order of IterVars in it. This is all the information we need to
 * tensorize a buffer access.
 */
struct RegionAndIVOrder {
  /*! \brief The initial buffer access region. Guaranteed to have trivial output-to-input dim
   * mapping. Never changes -- all dim mutation are recorded in other fields. */
  const BufferRegion region_;
  /*! \brief Stores the *original* order of IterVars. One entry per *input* dimension.
   * `iv_order.size() == in_n_dims`. The current order of IterVars can be derived when needed. */
  const std::vector<Optional<Var>> iv_order_;
  /*! \brief The current order of IterVars. One entry per *output* dimension. */
  std::vector<Var> current_iv_order_;

  RegionAndIVOrder() = delete;

  /*!
   * \brief Create a BufferRegion from the given `buffer` and access `indices`.
   * \note This function doesn't allow:
   * 1. multiple iter vars in a single index expression,
   * 2. multiple index dims using the same iter var.
   * Most of these cases cannot be expressed in tensor-block expressions at all. (Example:
   * convolution.)
   */
  static RegionAndIVOrder FromBufferAccess(const Buffer& buffer, const Array<PrimExpr>& indices,
                                           const std::vector<Var>& all_ivs,
                                           const Map<Var, Range>& vars_region,
                                           arith::Analyzer& analyzer);

  // Select the defined IterVars from `iv_order_`.
  std::vector<Var> SelectDefinedIVs() const {
    return support::filter_map(iv_order_, [](auto&& iv) { return iv; }).to_vector();
  }

  /*!
   * \brief Update the order of IterVars to `new_iv_order`.
   * \note `new_iv_order` is not required to contain all IterVars in `iv_order_`. These unspecified
   * IterVars will be left unmoved.
   */
  RegionAndIVOrder& UpdateToOrder(const std::vector<Var>& new_iv_order) {
    current_iv_order_ = new_iv_order;
    return *this;
  }
  RegionAndIVOrder& UpdateToOrder(const RegionAndIVOrder& other) {
    return UpdateToOrder(other.SelectDefinedIVs());
  }

  /*!
   * \brief Reconcile `iv_order_` and `current_iv_order_` to produce necessary information for
   * generating a BufferRegion.
   * \return A pair (new_order, out_to_in_dims). `new_order` is the combined IterVars from
   * `iv_order_` and `current_iv_order_`. `out_to_in_dims` is the mapping from output dimension to
   * input dimension used in BufferRegion.
   */
  std::pair<std::vector<Var>, Array<Optional<Integer>>> ReconcileIVOrders() const;

  //! \brief Resolve the region to a PrimExpr.
  PrimExpr Resolve() const;
};

template <typename Container>
auto keys(const Container& container) {
  return support::map(container, [](const auto& pair) { return pair.first; }).to_vector();
}

/*!
 * \brief A per-block re-tensorizer that converts a TVM block to a Triton-style (or say numpy-style)
 * tensorized expression.
 *
 * \example
 * Before:
 * \code
 * for i in range(10):
 *   for j in range(20):
 *     for k in range(30):
 *       vi, vj, vk = T.axis.remap("SSR", [i, j, k])
 *       with T.init():
 *         A[vi, vj] = T.float32(0.0)
 *       A[vi, vj] = A[vi, vj] + T.exp(B[vi, vj, vk] - C[vi, vj])
 * \endcode
 * After:
 * \code
 * A = tl.sum(tl.exp(B - C[:, :, None]), axis=2)
 * \endcode
 */
struct BlockRetensorizer : public ExprMutator {
  using ArangeBufferCallback = std::function<void(Buffer, PrimExpr)>;

  static Stmt Visit(const BlockRealize& br_to_visit, const std::vector<For>& outer_loops,
                    const std::vector<For>& inner_loops, bool keep_block_structure,
                    const ArangeBufferCallback& arange_buf_callback, arith::Analyzer& analyzer);

 private:
  BlockRetensorizer(const BlockRealize& br, const Map<Var, IterVar>& all_ivs,
                    const Map<Var, Range>& vars_region,
                    const ArangeBufferCallback& arange_buf_callback, arith::Analyzer& analyzer)
      : br_(br),
        all_ivs_(all_ivs),
        all_ivs_list_(keys(all_ivs)),
        vars_region_(vars_region),
        arange_buf_callback_(arange_buf_callback),
        analyzer_(analyzer),
        store_(Downcast<BufferStore>(br->block->body)) {
    auto store_region = CreateRegion(store_->buffer, store_->indices);
    store_region_ = Downcast<BufferRegion>(store_region.Resolve());
    store_iv_order_ =
        support::filter_map(store_region.iv_order_, [](auto&& iv) { return iv; }).to_vector();
  }

  /*! \brief Try to match a matmul pattern (2D implemented with `tl.dot` or 1D implemented with
   * `tl.sum`). Returns `std::nullopt` if no match. */
  std::optional<BufferRegionStore> TryMatchMatMulPattern(const Var& vk);
  /*! \brief Create a reduction operation. Any pattern mismatch will be a fatal error. */
  BufferRegionStore EmitReductionPattern(const std::vector<Var>& red_ivs);
  /*!
   \brief Create a map operation. Any pattern mismatch will be a fatal error.
   \param init_value Useful for emulating a 0D reduction block with a map pattern. A TVM reduction
   block can be 0D (doesn't have a reduction IterVar) but still has an init stmt, and EmitMapPattern
   handles this case. If `init_value` is provided, the block body `output[...] = f(output[...],
   ...)` will be substituted to `output[...] = f(init_value, ...)`.
   */
  BufferRegionStore EmitMapPattern(Optional<PrimExpr> init_value);

  RegionAndIVOrder CreateRegion(const Buffer& buffer, const Array<PrimExpr>& indices) {
    return RegionAndIVOrder::FromBufferAccess(buffer, indices, all_ivs_list_, vars_region_,
                                              analyzer_);
  }
  PrimExpr CreateRegionWithStoreOrder(const Buffer& buffer, const Array<PrimExpr>& indices) {
    return CreateRegion(buffer, indices).UpdateToOrder(store_iv_order_).Resolve();
  }

  /* Visitation methods */
  PrimExpr VisitExpr_(const BufferLoadNode* load) override {
    return CreateRegionWithStoreOrder(load->buffer, load->indices);
  }
  PrimExpr VisitExpr_(const CallNode* call_node) override;
  PrimExpr VisitExpr_(const VarNode* var_node) override;

  // Create a new buffer for arange(min, min + extent).
  Buffer CreateArangeBuffer(Range range);

  const BlockRealize& br_;
  const Map<Var, IterVar>& all_ivs_;
  const std::vector<Var> all_ivs_list_;
  const Map<Var, Range>& vars_region_;
  const ArangeBufferCallback& arange_buf_callback_;
  arith::Analyzer& analyzer_;
  const BufferStore store_;
  BufferRegion store_region_;
  std::vector<Var> store_iv_order_;
};

/*!
 * \brief Run re-tensorization at the top (full-function) level, applying the block-level
 * re-tensorizer to blocks with the right context.
 *
 * It assumes an "outer block -- inner block" structure established by `TileExprAutoBlockize`.
 * 1. An "innermost" block's body is a BufferStore. The block-level `BlockRetensorizer` will be
 * applied to innermost blocks.
 * 2. Each innermost block has an outer block. Together with loops, they form an `(outer_loops,
 * outer_block, inner_loops, inner_block)` structure. The inner block will be retensorized so that
 * all the `inner_loops` are contracted away.
 *
 * This pass creates the re-tensorized version of each innermost block, and returns a list of all
 * changes to be applied on the IR. The pass itself doesn't modify the IR. A later pass will carry
 * out the IR modifications.
 */
struct RetensorizerWithBlockScope : public StmtMutator {
  RetensorizerWithBlockScope(BlockMap* block_map) : block_map_(block_map) {}

  Stmt VisitStmt(const Stmt& stmt);

  Stmt VisitStmt_(const BlockRealizeNode* br_node);
  Stmt VisitStmt_(const BlockNode* block_node);

  BlockMap* block_map_;
  arith::Analyzer analyzer_;

  // The path from visitation start point to the current position.
  VisitorStack<Stmt> path_;
  // Map from old stmt to new stmt.
  Map<Stmt, Stmt> replace_map_;
  // New buffers to be added to each (outer) block.
  std::unordered_multimap<const BlockNode*, std::pair<Buffer, PrimExpr>> arange_buffers_;
};

/* Implementation details */

template <typename C1, typename C2>
bool set_equal(const C1& c1, const C2& c2) {
  using U1 = std::remove_cv_t<std::remove_reference_t<decltype(*c1.begin())>>;
  using U2 = std::remove_cv_t<std::remove_reference_t<decltype(*c2.begin())>>;
  static_assert(std::is_same_v<U1, U2>);
  std::unordered_set<U1> s1(c1.begin(), c1.end()), s2(c2.begin(), c2.end());
  return std::none_of(s1.begin(), s1.end(), [&](const U1& v) { return s2.count(v) == 0; }) &&
         std::none_of(s2.begin(), s2.end(), [&](const U1& v) { return s1.count(v) == 0; });
}

Stmt RetensorizerWithBlockScope::VisitStmt(const Stmt& stmt) {
  auto guard = path_.push(stmt);
  auto new_stmt = StmtMutator::VisitStmt(stmt);
  // 1. See if previous visitation has specified a replacement. This replacement does not compose
  // with the normal visitor mutation.
  auto it = replace_map_.find(stmt);
  if (it != replace_map_.end()) {
    ICHECK(new_stmt.same_as(stmt)) << "Replacement and mutation happening at the same time";
    return (*it).second;
  }
  // 2. Otherwise, return the normal visitor mutation.
  return new_stmt;
}

Stmt RetensorizerWithBlockScope::VisitStmt_(const BlockRealizeNode* br_node) {
  bool innermost = br_node->block->body->IsInstance<BufferStoreNode>();
  if (!innermost) {
    return Downcast<BlockRealize>(StmtMutator::VisitStmt_(br_node));
  }
  auto blk_name = br_node->block->name_hint;
  // If this block is an innermost block, list its inner loops and outer loops to prepare to call
  // `BlockRetensorizer`.
  const auto& data = path_.data();
  // data.rbegin() is ourselves.
  auto outer_br_it_rev = std::find_if(data.rbegin() + 1, data.rend(), [](const Stmt& stmt) {
    return stmt->IsInstance<BlockRealizeNode>();
  });
  ICHECK(outer_br_it_rev != data.rend())
      << "Block \"" << blk_name << "\" does not have an outer block: ";
  auto outer_block = Downcast<BlockRealize>(*outer_br_it_rev)->block;
  auto outer_br_it = data.begin() + std::distance(outer_br_it_rev, data.rend());
  auto ToFor = [](const Stmt& stmt) {
    auto for_ = stmt.as<For>();
    return for_.defined() ? for_.value() : std::optional<For>();
  };
  auto outer_loops = support::filter_map(data.begin(), outer_br_it, ToFor).to_vector();
  auto inner_loops = support::filter_map(outer_br_it + 1, data.end(), ToFor).to_vector();
  // Finally call `BlockRetensorizer`.
  bool keep_block_structure = block_map_ != nullptr;
  auto arange_buf_callback = [this, outer_block](Buffer buffer, PrimExpr expr) {
    arange_buffers_.insert({outer_block.get(), {buffer, expr}});
  };
  auto new_stmt = BlockRetensorizer::Visit(GetRef<BlockRealize>(br_node), outer_loops, inner_loops,
                                           keep_block_structure, arange_buf_callback, analyzer_);
  if (keep_block_structure) {
    block_map_->Insert(br_node->block, Downcast<BlockRealize>(new_stmt)->block);
  }
  // The insertion point for this block would be inner_loops[0], if that exists, otherwise the
  // inner BlockRealize itself.
  Stmt to_be_replaced = inner_loops.empty() ? GetRef<Stmt>(br_node) : inner_loops[0];
  replace_map_.Set(to_be_replaced, new_stmt);
  // Don't change the original block.
  return GetRef<Stmt>(br_node);
}

Stmt RetensorizerWithBlockScope::VisitStmt_(const BlockNode* block_node) {
  auto block = Downcast<Block>(StmtMutator::VisitStmt_(block_node));
  // NOTE: the key is the original block, not the mutated one.
  auto [it_begin, it_end] = arange_buffers_.equal_range(block_node);
  if (it_begin != it_end) {
    // If this block has new buffers to create, do that now. Add the buffer to the block's
    // alloc_buffers, and assign the expression to the buffer in the block's body.
    auto new_buffers = block->alloc_buffers;
    std::vector<Stmt> body_exprs;
    for (auto it = it_begin; it != it_end; ++it) {
      auto [buffer, arange_expr] = it->second;
      new_buffers.push_back(buffer);
      body_exprs.push_back(BufferRegionStore(BufferRegion::FullRegion(buffer), arange_expr));
    }
    arange_buffers_.erase(block_node);
    body_exprs.push_back(block->body);
    auto block_ptr = block.CopyOnWrite();
    block_ptr->alloc_buffers = new_buffers;
    block_ptr->body = SeqStmt(body_exprs);
  }
  if (block_map_) {
    block_map_->Insert(GetRef<Block>(block_node), block);
  }
  return block;
}

Optional<StringImm> GetReducerName(const CommReducer& reducer) {
  auto expr = reducer->result[0];
  if (expr->IsInstance<MaxNode>()) {
    return StringImm("max");
  } else if (expr->IsInstance<MinNode>()) {
    return StringImm("min");
  } else if (expr->IsInstance<AddNode>()) {
    return StringImm("sum");
  }
  return NullOpt;
}

Stmt BlockRetensorizer::Visit(const BlockRealize& br_to_visit, const std::vector<For>& outer_loops,
                              const std::vector<For>& inner_loops, bool keep_block_structure,
                              const ArangeBufferCallback& arange_buf_callback,
                              arith::Analyzer& analyzer) {
  // Add loop variables and their domains to `var_range`.
  Map<Var, Range> var_range;
  for (const auto& loop : outer_loops) {
    auto ann_orig_bounds = GetAnn<Range>(loop.get(), attr::tir_loop_original_bounds);
    auto range = ann_orig_bounds.defined() ? ann_orig_bounds.value()
                                           : Range::FromMinExtent(loop->min, loop->extent);
    var_range.Set(loop->loop_var, range);
  }
  // Look at the block's IterVars. Make a list of reduction IterVars, and a list of all IterVars.
  // Throw an error if any other IterVars are found.
  std::vector<Var> map_ivs, red_ivs;
  Map<Var, IterVar> all_ivs;
  for (const auto& iv : br_to_visit->block->iter_vars) {
    var_range.Set(iv->var, iv->dom);
    if (iv->iter_type == IterVarType::kCommReduce) {
      red_ivs.push_back(iv->var);
    } else {
      ICHECK(iv->iter_type == IterVarType::kDataPar) << "Unsupported IterVar type: " << iv;
      map_ivs.push_back(iv->var);
    }
    all_ivs.Set(iv->var, iv);
  }

  // Pattern matching for the BufferStore: map operation, reduction operation, matmul.
  BlockRetensorizer retensorizer(br_to_visit, std::move(all_ivs), var_range, arange_buf_callback,
                                 analyzer);
  ICHECK(set_equal(retensorizer.store_iv_order_, map_ivs))
      << "The set of spatial IterVars differ from the set of "
         "IterVars used in the LHS of the BufferStore: "
      << retensorizer.store_;
  auto new_stmt = [&red_ivs, &retensorizer, &br_to_visit]() {
    if (red_ivs.size() > 0) {
      if (red_ivs.size() == 1) {
        // 1. Matmul
        auto result = retensorizer.TryMatchMatMulPattern(red_ivs[0]);
        if (result.has_value()) {
          return result.value();
        }
      }
      return retensorizer.EmitReductionPattern(red_ivs);
    } else if (br_to_visit->block->init.defined()) {
      // There is this corner case where the block is a reduction (has init stmt) but is 0D.
      // We'll replace all reads from the write buffer with the init value, and emit a map pattern.
      auto init_stmt = Downcast<BufferStore>(br_to_visit->block->init);
      auto init_value = init_stmt->value;
      return retensorizer.EmitMapPattern(init_value);
    } else {
      return retensorizer.EmitMapPattern(NullOpt);
    }
  }();
  if (keep_block_structure) {
    // Make a dummy block and blockrealize with only the name.
    // TODO: compute read and write region.
    auto block = br_to_visit->block;
    return BlockRealize({}, const_true(), Block({}, {}, {}, block->name_hint, new_stmt));
  } else {
    return new_stmt;
  }
}

std::optional<BufferRegionStore> BlockRetensorizer::TryMatchMatMulPattern(const Var& vk) {
  // Match a 2D matmul of form
  // Out[vi, vj] = Acc[...] +
  //      Cast(Lhs[*batchB, *transposeB(... + vi, ... + vk)]) *
  //      Cast(Rhs[*batchC, *transposeC(... + vk, ... + vj)])
  // (Cast is optional, and transpose{B|C} could be identity or transposition.)
  // Or 1D matmul of form
  // Out[vi] = Acc[...] +
  //      Cast(Lhs[*batchB, ... + vk]) *
  //      Cast(Rhs[*batchC, *transposeC(... + vk, ... + vj)])
  using namespace arith;

  // 1. Match 3 BufferLoads from RHS, removing optional casts.
  BufferLoad acc, lhs, rhs;
  {
    PVar<BufferLoad> pacc, plhs, prhs;
    if (!(pacc + POptionalCast(plhs) * POptionalCast(prhs)).Match(store_->value)) {
      return std::nullopt;
    }
    std::tie(acc, lhs, rhs) = std::make_tuple(pacc.Eval(), plhs.Eval(), prhs.Eval());
  }
  // 2. Create and check the acc region. Only produce the acc argument if the block has no init
  // (in which case `out = dot(lhs, rhs, acc)`).
  // If the block has an init, we only require that the acc buffer is the output buffer.
  Optional<PrimExpr> acc_region = NullOpt;
  if (br_->block->init.defined()) {
    ICHECK(acc->buffer.same_as(store_->buffer));
  } else {
    acc_region = CreateRegionWithStoreOrder(acc->buffer, acc->indices);
  }
  // 3. Check if we're looking at a 2D matmul or a 1D matmul.
  auto& map_ivs = store_iv_order_;
  bool expect_mat_mat;
  std::vector<Var> lhs_ivs, rhs_ivs;
  if (store_iv_order_.size() == 2) {
    expect_mat_mat = true;
    lhs_ivs = {map_ivs[0], vk};
    rhs_ivs = {vk, map_ivs[1]};
  } else if (map_ivs.size() == 1) {
    expect_mat_mat = false;
    lhs_ivs = {vk};
    rhs_ivs = {map_ivs[0], vk};
  } else {
    return {};
  }
  // 4. Create regions from the matmul operands, and produce the matmul.
  auto lhs_region = CreateRegion(lhs->buffer, lhs->indices).UpdateToOrder(lhs_ivs).Resolve();
  auto rhs_region = CreateRegion(rhs->buffer, rhs->indices).UpdateToOrder(rhs_ivs).Resolve();
  auto out_dtype = store_->buffer->dtype;
  PrimExpr result;
  if (expect_mat_mat) {
    Array<PrimExpr> args({lhs_region, rhs_region});
    if (acc_region.defined()) {
      args.push_back(acc_region.value());
    }
    result = Call(out_dtype, tir::builtin::tile_dot(), args);
  } else {
    // tl.sum(cast(type2, lhs) * cast(type2, rhs)), dim=1)
    result = Call(
        out_dtype, tir::builtin::tile_reduce(),
        {cast(out_dtype, lhs_region) * cast(out_dtype, rhs_region), Integer(1), StringImm("sum")});
    if (acc_region.defined()) {
      result = acc_region.value() + result;
    }
  }
  // 6. Write the result to the buffer.
  return BufferRegionStore(store_region_, result);
}

BufferRegionStore BlockRetensorizer::EmitReductionPattern(const std::vector<Var>& red_ivs) {
  // Match a 1D reduction of form
  // Out[*axes1] = f(Out[*axes1], Expr)
  // `Expr` is a PrimExpr that must use the reduction axis (call it `vk`) of the block.
  // Therefore `In[vi, vj, vk]`, `In[vi, vk, vj]`, `In1[vi, vk] + In2[vi]` are all valid examples of
  // `Expr`.
  // - Multiple reduction itervars are not supported. The `tile_reduce` intrinsic only supports
  //   one reduction axis. If this is needed in the future, we can work around by
  //   emitting multiple reduction calls.
  ICHECK(red_ivs.size() == 1) << "Multiple reduction itervars are not supported";
  auto vk = red_ivs[0];

  // 1. Check for reduction pattern. The block must have a associative self-reduction:
  // `output[*indices] = f(output[*indices], ...)`
  auto red_match = tir::MatchSelfReduction(NullOpt, br_->block, 0);
  auto red_op_name = GetReducerName(red_match.reducer);
  ICHECK(red_op_name.defined()) << "Reduction must have a name";
  // 2. Determine the usage of `vk` in the RHS, and extract a single IV order to be used for this
  // block. This is tricky because there may be multiple `BufferLoad`s that use `vk` where the
  // relative order of `vk` differ.
  // We'll take the BufferLoad that uses `vk` and has the most unique IVs. This is a heuristic to
  // try and reduce the number of permutations.
  std::vector<Var> full_order;
  auto IsVK = [&](Optional<Var>&& expr) { return expr.same_as(vk); };
  {
    std::optional<RegionAndIVOrder> best_region;
    PostOrderVisit(red_match.rhs, [&](const ObjectRef& node) {
      if (auto load = node.as<BufferLoadNode>()) {
        auto this_region = CreateRegion(load->buffer, load->indices);
        auto ivs = this_region.SelectDefinedIVs();
        auto vk_index = std::find_if(ivs.begin(), ivs.end(), IsVK);
        if (vk_index != ivs.end() &&
            (!best_region || ivs.size() > best_region->SelectDefinedIVs().size())) {
          best_region.emplace(this_region);
        }
      }
    });
    ICHECK(best_region.has_value())
        << "RHS of reduction does not use the reduction axis of the block: " << red_match.rhs;
    // Reconcile the IV order of the best region with that of the store buffer.
    best_region->UpdateToOrder(store_iv_order_);
    full_order = best_region->ReconcileIVOrders().first;
  }
  auto it = std::find_if(full_order.begin(), full_order.end(), IsVK);
  ICHECK(it != full_order.end());
  size_t vk_index = std::distance(full_order.begin(), it);
  // 3. Create the region of RHS by recursively visiting, but make sure to use our new `iv_order`.
  store_iv_order_ = full_order;
  auto rhs_region = VisitExpr(red_match.rhs);
  // 4. Create the reduction call.
  auto out_dtype = store_->buffer->dtype;
  auto result = Call(out_dtype, tir::builtin::tile_reduce(),
                     {cast(out_dtype, rhs_region), Integer(vk_index), red_op_name.value()});
  // 5. Write the result to the buffer.
  return BufferRegionStore(store_region_, result);
}

BufferRegionStore BlockRetensorizer::EmitMapPattern(Optional<PrimExpr> init_value) {
  auto rhs = store_->value;
  if (init_value.defined()) {
    rhs = ReplaceBufferLoads(rhs, [&](const BufferLoadNode* load) {
      return load->buffer.same_as(store_->buffer) ? init_value.value() : GetRef<PrimExpr>(load);
    });
  }
  // Match a special case where RHS is a scalar.
  if (rhs->IsInstance<IntImmNode>() || rhs->IsInstance<FloatImmNode>()) {
    ICHECK(store_region_->IsFullRegion())
        << "Cannot fill a part of a buffer with a scalar. Block: " << br_;
    auto buffer = store_->buffer;
    String dtype = DLDataType2String(buffer->dtype);
    auto fill_args = Array<PrimExpr>({StringImm(dtype), rhs});
    for (auto& dim : buffer->shape) {
      fill_args.push_back(dim);
    }
    return BufferRegionStore(BufferRegion::FullRegion(buffer),
                             Call(buffer->dtype, tir::builtin::tile_full(), fill_args));
  }
  // Match an elementwise operation by simply recursively visiting the RHS and converting each
  // BufferLoad to a BufferRegion, each independent variable usage into a `tl.range(...)`, etc.
  return BufferRegionStore(store_region_, VisitExpr(rhs));
}

PrimExpr BlockRetensorizer::VisitExpr_(const CallNode* call_node) {
  if (call_node->op == tir::builtin::if_then_else()) {
    auto cond = VisitExpr(call_node->args[0]);
    auto then_case = VisitExpr(call_node->args[1]);
    auto else_case = VisitExpr(call_node->args[2]);
    return if_then_else(cond, then_case, else_case);
  }
  return ExprMutator::VisitExpr_(call_node);
}

PrimExpr BlockRetensorizer::VisitExpr_(const VarNode* var_node) {
  // No indices of buffer accesses are visited here. This must be a free-standing use of variable.
  auto var = GetRef<Var>(var_node);
  auto it = all_ivs_.find(var);
  if (it == all_ivs_.end()) {
    return var;
  }
  // Now this is an IterVar. Since we're tensorizing, we must replace the IterVar with something
  // tensor. Create a buffer to hold `range(min, min + extent)`.
  auto new_buffer = CreateArangeBuffer((*it).second->dom);
  // Pretend we have a `new_buffer[var]` access. CreateRegion should take care of the rest.
  return CreateRegionWithStoreOrder(new_buffer, {var});
}

Buffer BlockRetensorizer::CreateArangeBuffer(Range range) {
  auto dtype = range->min->dtype;
  auto min = as_const_int(range->min), extent = as_const_int(range->extent);
  ICHECK(min && extent) << "tile_arange bounds must be constant";
  String new_buf_name = "arange_" + std::to_string(*min) + "_" + std::to_string(*extent);
  // TODO: can we not hardcode "shared" here?
  auto new_buffer = decl_buffer({Integer(*extent)}, dtype, new_buf_name, "shared");
  auto arange_expr =
      Call(dtype, tir::builtin::tile_arange(), {range->min, range->min + range->extent});
  arange_buf_callback_(new_buffer, arange_expr);
  return new_buffer;
}

RegionAndIVOrder RegionAndIVOrder::FromBufferAccess(const Buffer& buffer,
                                                    const Array<PrimExpr>& indices,
                                                    const std::vector<Var>& all_ivs,
                                                    const Map<Var, Range>& vars_region,
                                                    arith::Analyzer& analyzer) {
  // 1. Run subspace divide where the iter vars are the inner region.
  auto [marks, _] = arith::SubspaceDivide(indices, vars_region, all_ivs, const_true(), analyzer);
  // 2. Make ranges for the BufferRegion.
  Array<Range> ranges;
  for (const auto& [outer, inner] : marks) {
    auto outer_expr = arith::NormalizeIterMapToExpr(outer->source);
    ranges.push_back(Range::FromMinExtent(outer_expr * inner->extent, inner->extent));
  }
  // 3. Check that each mark's inner split only uses 1 iter var, and find that iter var.
  std::vector<Optional<Var>> used_ivs;
  std::vector<std::optional<size_t>> out_to_in_dims;
  for (size_t mark_dim = 0; mark_dim < marks.size(); ++mark_dim) {
    auto& [outer, inner] = marks[mark_dim];
    auto inner_expr = arith::NormalizeIterMapToExpr(inner->source);
    auto CheckCond = [&](bool cond, const std::string& error_msg) {
      ICHECK(cond) << "An inner split of an indexing expression " << error_msg
                   << ": split = " << inner_expr << ", the original index expr is "
                   << indices[mark_dim];
    };
    Optional<Var> iv_var{};
    PostOrderVisit(inner_expr, [&](const ObjectRef& obj) {
      if (auto var = obj.as<VarNode>()) {
        if (iv_var.defined()) {
          CheckCond(iv_var.get() == var, "uses multiple IterVars");
        } else {
          iv_var = GetRef<Var>(var);
        }
      }
    });
    used_ivs.push_back(iv_var);
    if (iv_var.defined()) {
      out_to_in_dims.push_back(mark_dim);
    }
  }
  auto result = RegionAndIVOrder{.region_ = BufferRegion(buffer, ranges), .iv_order_ = used_ivs};
  result.UpdateToOrder(result);
  return result;
}

template <typename T1, typename Func>
std::vector<size_t> StableArgsortOptional(const std::vector<T1>& xs, Func&& key_f) {
  using OptT2 = std::invoke_result_t<Func, T1>;
  // Argsort the non-null values first.
  std::vector<OptT2> ys;
  std::vector<size_t> indices;
  for (size_t i = 0; i < xs.size(); ++i) {
    auto key = key_f(xs[i]);
    ys.push_back(key);
    if (key.has_value()) {
      indices.push_back(i);
    }
  }
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { return *ys[a] < *ys[b]; });
  // Then fill in the indices of the None values.
  std::vector<size_t> result(xs.size());
  size_t idx_ptr = 0;
  for (size_t i = 0; i < ys.size(); ++i) {
    result[i] = ys[i].has_value() ? indices[idx_ptr++] : i;
  }
  return result;
}

template <typename T1, typename Func>
std::vector<T1> StableSortOptional(const std::vector<T1>& xs, Func&& key_f) {
  auto argsort = StableArgsortOptional(xs, std::forward<Func>(key_f));
  std::vector<T1> result(xs.size());
  for (size_t i = 0; i < argsort.size(); ++i) {
    result[i] = xs[argsort[i]];
  }
  return result;
}

template <typename K, typename V>
std::optional<V> Get(const std::unordered_map<K, V>& dict, const K& key) {
  auto it = dict.find(key);
  return it == dict.end() ? std::optional<V>() : it->second;
}

std::pair<std::vector<Var>, Array<Optional<Integer>>> RegionAndIVOrder::ReconcileIVOrders() const {
  const auto& xs = iv_order_;
  const auto& ys = current_iv_order_;
  // First obtain a sorted version of `ys` in `xs` order. This is so that we can match xs and ys
  // more easily. `ys` elements that are not in `xs` are left in their original position.
  std::unordered_map<Var, size_t> xs_indices;
  for (size_t i = 0; i < xs.size(); ++i) {
    if (xs[i].defined()) {
      xs_indices.emplace(xs[i].value(), i);
    }
  }
  auto ys_sorted = StableSortOptional(ys, [&](auto&& x) { return Get(xs_indices, x); });
  // Then start a two-pointer scan to incorporate xs and ys variables into one vector.
  std::vector<Var> full_order;
  Array<Optional<Integer>> o2i_dims;
  size_t xs_idx = 0, ys_idx = 0;
  while (xs_idx < xs.size() && ys_idx < ys.size()) {
    auto y = ys_sorted[ys_idx];
    if (y.same_as(xs[xs_idx])) {
      // Case 1. `y` is in both `xs` and `ys` (and matching).
      full_order.push_back(y);
      o2i_dims.push_back(Integer(xs_idx));
      ++xs_idx;
      ++ys_idx;
    } else if (xs_indices.count(y)) {
      // Case 2. If `y` is in `xs`, but `y != xs[xs_idx]`, then `xs[xs_idx]` is not in `ys`.
      if (xs[xs_idx].defined()) {
        full_order.push_back(xs[xs_idx].value());
        o2i_dims.push_back(Integer(xs_idx));
      }
      ++xs_idx;
    } else {
      // Case 3. `y` is not in `xs`.
      full_order.push_back(y);
      o2i_dims.push_back(NullOpt);
      ++ys_idx;
    }
  }
  for (; xs_idx < xs.size(); ++xs_idx) {
    if (xs[xs_idx].defined()) {
      full_order.push_back(xs[xs_idx].value());
      o2i_dims.push_back(Integer(xs_idx));
    }
  }
  for (; ys_idx < ys.size(); ++ys_idx) {
    full_order.push_back(ys[ys_idx]);
    o2i_dims.push_back(NullOpt);
  }
  return {full_order, o2i_dims};
}

PrimExpr RegionAndIVOrder::Resolve() const {
  // Task: match our `iv_order_`(xs) to `current_iv_order_`(ys), to produce an
  // `out = vector<optional<size_t>>`.
  auto [new_iv_order, out_to_in_dims] = ReconcileIVOrders();
  auto new_region = BufferRegion(region_->buffer, region_->region, out_to_in_dims);
  // Now we just need to sort `new_iv_order` to match `current_iv_order_`, and this will exactly
  // produce the permutation.
  auto ys_indices = support::enumerate</*IdxFirst=*/false>(current_iv_order_)
                        .to_container<std::unordered_map<Var, size_t>>();
  auto argsort = StableArgsortOptional(new_iv_order, [&](auto&& x) { return Get(ys_indices, x); });
  bool is_identity = std::is_sorted(argsort.begin(), argsort.end());
  if (!is_identity) {
    Array<PrimExpr> args({new_region});
    for (size_t i = 0; i < argsort.size(); ++i) {
      args.push_back(Integer(argsort[i]));
    }
    return Call(new_region->dtype, tir::builtin::tile_permute(), args);
  }
  return new_region;
}

}  // namespace

Stmt TensorizeIntoTileExpr(Stmt body, BlockMap* block_map) {
  return RetensorizerWithBlockScope(block_map)(body);
}

}  // namespace tvm::tir
