/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file compact_buffer_region.cc
 * \brief Compact the buffer size into its exact need.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/utils.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/const_fold.h"
#include "../../support/nd_int_set.h"
#include "../schedule/analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

/*! \brief a more constrained bound estimate for n-dimentional int set */
NDIntSet NDIntSetEval(Region region, PrimExpr predicate,
                      const std::unordered_map<const VarNode*, arith::IntSet>& dom_map,
                      arith::Analyzer* analyzer) {
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectEqual> var_dom;
  for (const auto& it : dom_map) {
    var_dom[GetRef<Var>(it.first)] = it.second.CoverRange(Range::FromMinExtent(0, 0));
  }
  Optional<Array<arith::IntSet>> eval_res =
      arith::EstimateRegionUpperBound(region, var_dom, predicate, analyzer);

  if (eval_res.defined()) {
    return NDIntSet(eval_res.value().begin(), eval_res.value().end());
  }
  return support::NDIntSetEval(support::NDIntSetFromRegion(region), dom_map);
}

/*!
 * \brief Collect buffer aliasing information.
 */
class Var2BufferCollector : public StmtExprVisitor {
 public:
  /*! \brief Map the buffer var to all aliased buffers. */
  std::unordered_map<Var, std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>> var2buffer_;

 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    var2buffer_[op->buffer->data].insert(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    var2buffer_[op->buffer->data].insert(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    for (const Buffer& buffer : op->alloc_buffers) {
      var2buffer_[buffer->data].insert(buffer);
    }
    for (const MatchBufferRegion& region : op->match_buffers) {
      var2buffer_[region->buffer->data].insert(region->buffer);
      var2buffer_[region->source->buffer->data].insert(region->source->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode* op) final {
    var2buffer_[op->buffer->data].insert(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*!
 * \brief Collect the access region of each buffer.
 * \note The param buffer regions will not be collected.
 */
class BufferAccessRegionCollector : public StmtExprVisitor {
 public:
  static std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> Collect(
      const PrimFunc& f, bool collect_inbound) {
    BufferAccessRegionCollector region_collector(collect_inbound);

    // collect buffer var to aliased buffer mapping
    Var2BufferCollector var2buffer_collector;
    var2buffer_collector(f->body);
    std::swap(region_collector.var2buffer_, var2buffer_collector.var2buffer_);

    // collect buffer access regions
    region_collector(f->body);
    return std::move(region_collector.buffer_access_region_);
  }

 private:
  struct BufferAccessInfo {
    /*! \brief The buffer. */
    Buffer buffer;
    /*! \brief The buffer access region, which can be updated during visiting. */
    NDIntSet accessed_region;

    explicit BufferAccessInfo(const Buffer& buffer, const NDIntSet& region)
        : buffer(buffer), accessed_region(region) {}
  };

  explicit BufferAccessRegionCollector(bool collect_inbound) : collect_inbound_(collect_inbound) {}

  /**************** Visitor overload ****************/

  void VisitStmt_(const BufferStoreNode* op) final {
    VisitBufferAccess(BufferRegion::FromPoint(op->buffer, op->indices));
    VisitExpr(op->value);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    VisitBufferAccess(BufferRegion::FromPoint(op->buffer, op->indices));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) final { VisitBufferVar(GetRef<Var>(op)); }

  void VisitStmt_(const ForNode* op) final {
    auto ann_orig_bounds = GetAnn<Range>(op, attr::tir_loop_original_bounds);
    auto loop_range = ann_orig_bounds.defined() ? ann_orig_bounds.value()
                                                : Range::FromMinExtent(op->min, op->extent);
    IterVar iter = op->kind == ForKind::kThreadBinding
                       ? IterVar(Range(), op->loop_var, IterVarType::kThreadIndex,
                                 op->thread_binding.value()->thread_tag)
                       : IterVar(Range(), op->loop_var, IterVarType::kDataPar);
    ancestor_iters_.push_back(iter);
    dom_analyzer_.Bind(op->loop_var, loop_range);
    dom_map_.emplace(op->loop_var.get(), arith::IntSet::FromRange(loop_range));
    StmtExprVisitor::VisitStmt_(op);
    dom_map_.erase(op->loop_var.get());
    ancestor_iters_.pop_back();
  }

  void VisitStmt_(const LetStmtNode* op) final {
    StmtExprVisitor::VisitExpr(op->value);
    if (arith::IsIndexType(op->value->dtype)) {
      dom_analyzer_.Bind(op->var, op->value);
      dom_map_.emplace(op->var.get(), arith::IntSet::SinglePoint(op->value));
    }
    StmtExprVisitor::VisitStmt(op->body);
    if (arith::IsIndexType(op->value->dtype)) {
      dom_map_.erase(op->var.get());
    }
  }

  void VisitExpr_(const LetNode* op) final {
    StmtExprVisitor::VisitExpr(op->value);
    if (arith::IsIndexType(op->value->dtype)) {
      dom_analyzer_.Bind(op->var, op->value);
      dom_map_.emplace(op->var.get(), arith::IntSet::SinglePoint(op->value));
    }
    StmtExprVisitor::VisitExpr(op->body);
    if (arith::IsIndexType(op->value->dtype)) {
      dom_map_.erase(op->var.get());
    }
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    // Visit condition
    StmtExprVisitor::VisitExpr(op->condition);
    {
      // Visit then branch
      With<ConditionalBoundsContext> ctx(op->condition, &dom_map_, &hint_map_,
                                         &pending_conditions_);
      StmtExprVisitor::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      // Visit else branch
      With<ConditionalBoundsContext> ctx(!op->condition, &dom_map_, &hint_map_,
                                         &pending_conditions_);
      StmtExprVisitor::VisitStmt(op->else_case.value());
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      // Visit condition
      StmtExprVisitor::VisitExpr(op->args[0]);
      {
        // Visit then branch
        With<ConditionalBoundsContext> ctx(op->args[0], &dom_map_, &hint_map_,
                                           &pending_conditions_);
        StmtExprVisitor::VisitExpr(op->args[1]);
      }
      {
        // Visit else branch
        With<ConditionalBoundsContext> ctx(!op->args[0], &dom_map_, &hint_map_,
                                           &pending_conditions_);
        StmtExprVisitor::VisitExpr(op->args[2]);
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    // Step 1. Record and update current read/write region annotations
    std::unordered_map<Buffer, std::vector<BufferRegion>, ObjectPtrHash, ObjectPtrEqual>
        cur_access_annotations;
    for (const BufferRegion& region : op->reads) {
      cur_access_annotations[region->buffer].push_back(region);
    }
    for (const BufferRegion& region : op->writes) {
      cur_access_annotations[region->buffer].push_back(region);
    }
    for (auto& [buf, regions] : cur_access_annotations) {
      regions.swap(access_annotations_[buf]);
    }
    // Step 2. Record relax position of ancestor_loops_
    for (const Buffer& buffer : op->alloc_buffers) {
      VisitBufferDef(buffer->data);
    }
    // Step 3. Visit match buffers
    for (const MatchBufferRegion& region : op->match_buffers) {
      VisitBufferAccess(region->source);
    }
    // Step 4. Visit block body recursively
    StmtExprVisitor::VisitStmt_(op);
    // Step 5. Recover read/write region annotations
    for (auto& p : cur_access_annotations) {
      auto& regions = access_annotations_[p.first];
      if (p.second.empty()) {
        access_annotations_.erase(p.first);
      } else {
        regions.swap(p.second);
      }
    }
    // Step 6. Update buffer_access_region_ from relaxed_accesses_ for inner buffers.
    for (const Buffer& buffer : op->alloc_buffers) {
      ICHECK_EQ(var2buffer_[buffer->data].size(), 1)
          << "Block allocation buffer shoud not be alised";
      SimplifyAndNarrowBufferRegionFromNDIntSet(buffer);
    }
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    With<ConditionalBoundsContext> ctx(op->predicate, &dom_map_, &hint_map_, &pending_conditions_);
    binding_ = GetBindings(GetRef<BlockRealize>(op));
    StmtExprVisitor::VisitStmt_(op);
    binding_.clear();
  }

  void VisitStmt_(const AllocateNode* op) final {
    auto it = var2buffer_.find(op->buffer_var);

    // Do not make compaction when the buffer def and
    // the allocation is not one-to-one with the same dtype.
    if (it == var2buffer_.end() || it->second.size() > 1) {
      return StmtExprVisitor::VisitStmt_(op);
    }
    const Buffer& buffer = *it->second.begin();
    if (buffer->dtype != op->dtype) {
      return StmtExprVisitor::VisitStmt_(op);
    }

    // Step 0. Record relax position of ancestor_loops_
    VisitBufferDef(op->buffer_var);
    // Step 1. Visit block body recursively
    StmtExprVisitor::VisitStmt(op->body);
    // Step 2. Update buffer_access_region_ from relaxed_accesses_ for inner buffers.
    SimplifyAndNarrowBufferRegionFromNDIntSet(buffer);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      IterVar iter = Downcast<IterVar>(op->node);
      ancestor_iters_.push_back(iter);
      Range dom = iter->dom;
      if (!dom.defined()) {  // dom is empty for legacy te schedule
        dom = Range::FromMinExtent(make_zero(op->value->dtype), op->value);
      }
      dom_analyzer_.Bind(iter->var, dom);
      dom_map_.emplace(iter->var.get(), arith::IntSet::FromRange(dom));
      StmtExprVisitor::VisitStmt_(op);
      dom_map_.erase(iter->var.get());
      ancestor_iters_.pop_back();
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /**************** Helper functions ****************/

  /*! \brief Record information on the buffer defining point. */
  void VisitBufferDef(const Var& buffer_data) {
    auto it = buffer_scope_depth_.find(buffer_data);
    ICHECK(it == buffer_scope_depth_.end()) << buffer_data << " has duplicate definitions";
    buffer_scope_depth_.insert(it, {buffer_data, ancestor_iters_.size()});
  }

  void VisitBufferAccess(const BufferRegion& buffer_region) {
    const Buffer& buffer = buffer_region->buffer;
    auto it = buffer_scope_depth_.find(buffer->data);
    if (it != buffer_scope_depth_.end()) {
      size_t n_ancestor_loops = it->second;
      // Step 1. Stop ancestor loop vars out of the allocation block from
      // being relaxed unless NeedRelaxThread() is true.
      std::vector<arith::IntSet> non_relaxed(n_ancestor_loops);
      for (size_t i = 0; i < n_ancestor_loops; ++i) {
        const IterVar& iter = ancestor_iters_[i];
        const VarNode* v = iter->var.get();
        if (NeedRelaxThread(iter, runtime::StorageScope::Create(buffer.scope()))) {
          continue;
        }
        auto dom_it = dom_map_.find(v);
        ICHECK(dom_it != dom_map_.end())
            << "Could not find domain for loop variable " << v->name_hint;
        non_relaxed[i] = dom_it->second;
        dom_map_.erase(dom_it);
      }
      // Step 2. Relax the access region
      auto normalize_pred = [](const PrimExpr& pred) {
        if (pred->dtype.is_bool()) return pred;
        return pred != make_zero(pred->dtype);
      };
      PrimExpr predicate = dom_analyzer_.Simplify(
          std::accumulate(pending_conditions_.begin(), pending_conditions_.end(), const_true(),
                          [normalize_pred](const PrimExpr& x, const PrimExpr& y) {
                            return normalize_pred(x) && normalize_pred(y);
                          }));
      Region region = Substitute(buffer_region->region, binding_);
      NDIntSet nd_int_set = NDIntSetEval(region, predicate, dom_map_, &dom_analyzer_);

      // Step 3. Restore the non-relaxed ancestor loops domain
      for (size_t i = 0; i < n_ancestor_loops; ++i) {
        const VarNode* v = ancestor_iters_[i]->var.get();
        dom_map_.emplace(v, non_relaxed[i]);
      }
      // Step 4. Update relaxed_accesses_ dict
      auto access_it = relaxed_accesses_.find(buffer);
      if (access_it != relaxed_accesses_.end()) {
        support::NDIntSetUnionWith(&access_it->second, nd_int_set);
      } else {
        relaxed_accesses_.insert(access_it, {buffer, nd_int_set});
      }
    }
  }

  void VisitBufferVar(const Var& var) {
    auto it = var2buffer_.find(var);
    if (it == var2buffer_.end()) {
      return;
    }
    for (const Buffer& buffer : it->second) {
      auto annotation_it = access_annotations_.find(buffer);
      if (annotation_it != access_annotations_.end()) {
        // opaque buffer has explicit accessed region annotations
        for (const BufferRegion& region : annotation_it->second) {
          VisitBufferAccess(region);
        }
      } else {
        VisitBufferAccess(BufferRegion::FullRegion(buffer));
      }
    }
  }

  /*! \brief Check whether the thread binding iter should be relaxed with given storage scope. */
  static bool NeedRelaxThread(const IterVar& iter, const runtime::StorageScope& scope) {
    if (iter->iter_type != IterVarType::kThreadIndex) {
      return false;
    }
    ICHECK(iter->thread_tag.defined());
    // When there is warp memory
    // threadIdx.x must be set to be warp index.
    return CanRelaxStorageUnderThread(scope, runtime::ThreadScope::Create((iter->thread_tag)));
  }

  /*!
   * \brief simplify and narrow down the region collected by NDIntSet.
   * Update the `relaxed_accesses_` dict. If `collect_inbound_` is true,
   * the result region would never exceed the original buffer shape.
   */
  void SimplifyAndNarrowBufferRegionFromNDIntSet(const Buffer& buffer) {
    auto it = relaxed_accesses_.find(buffer);
    ICHECK(it != relaxed_accesses_.end())
        << buffer << " is allocated but not accessed within block scope";

    const Array<PrimExpr>& original_shape = buffer->shape;
    const NDIntSet& nd_int_set = it->second;
    Array<Range>& result_region = buffer_access_region_[buffer];
    result_region.resize(nd_int_set.size());

    for (size_t i = 0; i < nd_int_set.size(); ++i) {
      const arith::IntSet& int_set = nd_int_set[i];
      Range original =
          Range(/*begin=*/make_zero(original_shape[i]->dtype), /*end=*/original_shape[i]);
      Range range = int_set.CoverRange(original);
      PrimExpr min, extent;
      if (collect_inbound_) {
        min = dom_analyzer_.Simplify(tvm::max(0, range->min));
        extent = range->extent;
        // Apply stronger symbolic proof to help us remove symbolic min here.
        if (!dom_analyzer_.CanProveLessEqualThanSymbolicShapeValue(extent, original_shape[i])) {
          extent = tvm::min(original_shape[i], range->extent);
        }
        extent = dom_analyzer_.Simplify(extent);
      } else {
        min = dom_analyzer_.Simplify(range->min);
        extent = dom_analyzer_.Simplify(range->extent);
      }

      // We check the buffer extent is pure and not loop dependent, since loop dependent
      // or data dependent allocation is not supported yet. Otherwise we should
      // fallback to use original buffer shape.
      if (SideEffect(extent) > CallEffectKind::kPure) {
        result_region.Set(i, original);
        continue;
      }
      auto is_loop_var = [this](const VarNode* v) {
        return std::any_of(ancestor_iters_.begin(), ancestor_iters_.end(),
                           [v](const IterVar& n) { return n->var.get() == v; });
      };
      if (UsesVar(extent, is_loop_var)) {
        // try estimate a constant upperbound on region's extent
        int64_t upperbound = dom_analyzer_.const_int_bound(extent)->max_value;
        if (upperbound != arith::ConstIntBound::kPosInf) {
          extent = make_const(extent->dtype, upperbound);
        } else {
          result_region.Set(i, original);
          continue;
        }
      }
      result_region.Set(i, Range::FromMinExtent(min, extent));
    }
  }

  /**************** Class members ****************/
  /*! \brief Only collect accessed region within original buffer shape bound. */
  bool collect_inbound_{true};

  /*! \brief The iteration scopes from the current node up to the root. */
  std::vector<IterVar> ancestor_iters_;

  /*!
   * \brief Map each buffer var to the n_ancester_loop. which is the loop depth at the
   * define point. ancestor_loops_[0: n_ancester_loop] should not be relaxed when
   * we evaluate this buffer's access regions.
   */
  std::unordered_map<Var, size_t> buffer_scope_depth_;

  /*! \brief Map the buffer var to all aliased buffers. */
  std::unordered_map<Var, std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>> var2buffer_;

  /*! \brief The map from loop vars to their iter range. */
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  /*! \brief Extra map from free vars to their iter range hints. */
  std::unordered_map<const VarNode*, arith::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief The analyzer aware of loop domains. */
  arith::Analyzer dom_analyzer_;
  /*! \brief The map from Buffer to it's relaxed access set. */
  std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> relaxed_accesses_;

  /*!
   * \brief The map from Buffer to it entire access region, used for returning.
   * The entire access region should get updated on the buffer's define point
   * and we sanity check that every buffer is defined only once.
   */
  std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> buffer_access_region_;

  /*! \brief The map from Buffer to it's access regions annotated by current block. */
  std::unordered_map<Buffer, std::vector<BufferRegion>, ObjectPtrHash, ObjectPtrEqual>
      access_annotations_;

  /*! \brief A mapping from variable of an IterVar to its expr. */
  Map<Var, PrimExpr> binding_;
};

/*! \brief The storage alignment for a dimension */
struct DimAlignInfo {
  /*! \brief The factor of the alignment */
  int align_factor{0};
  /*! \brief The offset of the alignment */
  int align_offset{0};
};

Array<PrimExpr> CalcStrides(const std::vector<DimAlignInfo>& dim_aligns,
                            const Array<PrimExpr>& shape) {
  std::vector<PrimExpr> strides;
  if (dim_aligns.size()) {
    ICHECK(dim_aligns.size() == shape.size());
    strides.resize(shape.size());
    PrimExpr stride = make_const(shape[0].dtype(), 1);
    for (size_t i = shape.size(); i != 0; --i) {
      size_t dim = i - 1;
      DimAlignInfo info = dim_aligns[dim];
      int align_factor = info.align_factor;
      int align_offset = info.align_offset;
      if (align_factor != 0) {
        PrimExpr factor = make_const(stride.dtype(), align_factor);
        PrimExpr offset = make_const(stride.dtype(), align_offset);
        stride = stride + indexmod(factor + offset - indexmod(stride, factor), factor);
      }
      strides[dim] = stride;
      stride = stride * shape[dim];
    }
  }
  return strides;
}

struct BufferAllocInfo {
  /*! \brief The buffer access region. */
  Region region;
  /*! \brief The storage alignment information. */
  std::vector<DimAlignInfo> dim_aligns;
  /*!
   * \brief The reallocated buffer with minimal size.
   * \note The value if NullOpt if the buffer do not need reallocate (e.g parameter buffer).
   */
  Buffer new_buffer;
  /*! \brief The dimensions that don't have size-1 extent. */
  std::vector<int> non_trivial_dims;
  /*! \brief Number of dimensions in the old buffer (before removing trivial dims, if any) */
  size_t prev_n_dim;

  BufferAllocInfo(Region region, Buffer old_buffer, std::vector<int> non_trivial_dims,
                  const std::optional<StorageAlignAnnotation>& storage_align)
      : non_trivial_dims(std::move(non_trivial_dims)), prev_n_dim(old_buffer->shape.size()) {
    // Remove size-1 dimensions from `region`. `shape` and `strides` will follow.
    this->region = RemoveTrivialDims(region);
    // Set dim alignment info
    if (storage_align.has_value()) {
      this->dim_aligns.resize(prev_n_dim);
      for (const StorageAlignTuple& dim_align : *storage_align) {
        ICHECK(dim_align.size() == 4);
        int dim = dim_align[1]->value;
        int factor = dim_align[2]->value;
        int offset = dim_align[3]->value;
        this->dim_aligns.at(dim) = {factor, offset};
      }
      // Remove size-1 dimensions from `dim_aligns`.
      // FIXME: if someone is actually padding trivial (size-1) dims into a non-1 size, this will
      // break.
      this->dim_aligns = RemoveTrivialDims(this->dim_aligns);
    }
    // Create new buffer
    Array<PrimExpr> shape = this->region.Map([](const Range& range) { return range->extent; });
    Array<PrimExpr> strides = CalcStrides(this->dim_aligns, shape);
    ObjectPtr<BufferNode> n = make_object<BufferNode>(*old_buffer.get());
    n->shape = std::move(shape);
    n->strides = std::move(strides);
    this->new_buffer = Buffer(n);
  }

  template <typename Container>
  Container RemoveTrivialDims(const Container& arr) const {
    Container res;
    for (size_t i = 0; i < non_trivial_dims.size(); ++i) {
      res.push_back(arr[non_trivial_dims[i]]);
    }
    return res;
  }

  /*! \brief Map from old buffer dimensions to new buffer dimensions. */
  auto MakeDimRemapping() const {
    std::vector<std::optional<int>> ret(prev_n_dim, std::nullopt);
    for (size_t i = 0; i < non_trivial_dims.size(); ++i) {
      ret[non_trivial_dims[i]] = i;
    }
    return ret;
  }
};

/*! \brief Reallocate the buffers with minimal region. */
class BufferCompactor : public StmtExprMutator {
 public:
  explicit BufferCompactor(std::unordered_map<Var, BufferAllocInfo> buffer_info,
                           BlockMap* block_map)
      : buffer_info_(std::move(buffer_info)), block_map_(block_map) {}

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    BufferStoreNode* op = store.CopyOnWrite();
    RewriteBufferAccess(op->buffer, op->indices);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    BufferLoadNode* op = load.CopyOnWrite();
    RewriteBufferAccess(op->buffer, op->indices);
    return std::move(load);
  }

  Stmt VisitStmt_(const BlockNode* op) final { return VisitBlock(op); }

  Block VisitBlock(const BlockNode* op) {
    // Step 1. Reallocate and rewrite alloc_buffers, also update BufferAllocInfo.
    Array<Buffer> alloc_buffers =
        op->alloc_buffers.Map([this](const Buffer& buf) { return RewriteAllocBuffer(buf); });
    // Step 2. Recursively rewrite BufferLoad/BufferStore.
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    // Step 3. Update block signature.
    {
      BlockNode* n = block.CopyOnWrite();
      auto RewriteBufferRegion_ = [this](const BufferRegion& buf_region) {
        return RewriteBufferRegion(buf_region);
      };
      auto RewriteMatchBuffer_ = [this](const MatchBufferRegion& match_buffer) {
        return MatchBufferRegion(match_buffer->buffer, RewriteBufferRegion(match_buffer->source));
      };
      n->reads = n->reads.Map(RewriteBufferRegion_);
      n->writes = n->writes.Map(RewriteBufferRegion_);
      n->match_buffers = n->match_buffers.Map(RewriteMatchBuffer_);
      n->alloc_buffers = std::move(alloc_buffers);
    }
    if (block_map_) {
      block_map_->Insert(GetRef<Block>(op), block);
    }
    return std::move(block);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Buffer new_buffer = RewriteAllocBuffer(op->buffer);
    auto n = CopyOnWrite(op);
    n->buffer = std::move(new_buffer);
    n->body = VisitStmt(op->body);
    return DeclBuffer(n);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    Allocate allocate = Downcast<Allocate>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_info_.find(allocate->buffer_var);
    if (it == buffer_info_.end()) {
      return std::move(allocate);
    }
    // Rewrite allocation shape if the corresponding buffer is in the buffer_info_
    // dict and the dtype is consistent, which denotes there are no buffer aliasing
    // and the compaction is safe.
    const Buffer& new_buffer = it->second.new_buffer;
    if (op->dtype != new_buffer->dtype) {
      return std::move(allocate);
    }
    Array<PrimExpr> new_shape = GetBufferAllocationShape(new_buffer);
    auto n = allocate.CopyOnWrite();
    ICHECK(n->buffer_var.same_as(new_buffer->data));
    n->extents = new_shape;
    return std::move(allocate);
  }

  Buffer RewriteAllocBuffer(const Buffer& buffer) {
    auto it = buffer_info_.find(buffer->data);
    if (it != buffer_info_.end()) {
      return it->second.new_buffer;
    }
    return buffer;
  }

  void RewriteBufferAccess(Buffer& buffer, Array<PrimExpr>& indices) {
    auto it = buffer_info_.find(buffer->data);
    if (it == buffer_info_.end()) {
      return;
    }
    const BufferAllocInfo& info = it->second;
    indices = RewriteBufferIndices(buffer, indices, info);
    buffer = info.new_buffer;
  }

  BufferRegion RewriteBufferRegion(const BufferRegion& buffer_region) {
    auto buffer = buffer_region->buffer;
    auto it = buffer_info_.find(buffer->data);
    if (it == buffer_info_.end()) {
      return buffer_region;
    }
    const BufferAllocInfo& info = it->second;
    Array<PrimExpr> region_min, region_extents;
    for (const Range& range : buffer_region->region) {
      region_min.push_back(range->min);
      region_extents.push_back(range->extent);
    }
    // RewriteBufferAccess removes trivial dims of `region_min`.
    region_min = RewriteBufferIndices(buffer, region_min, info);
    region_extents = info.RemoveTrivialDims(region_extents);
    // Remove trivial dims of `output_to_input_dims`.
    Array<Optional<Integer>> new_o2i;
    auto non_trivial_dims = info.MakeDimRemapping();
    for (const auto& dim : buffer_region->output_to_input_dims) {
      if (!dim.defined()) {
        new_o2i.push_back(dim);
        continue;
      }
      auto in_dim = non_trivial_dims[dim.value()->value];
      if (in_dim.has_value()) {
        new_o2i.push_back(Integer(in_dim.value()));
      }
    }
    buffer = info.new_buffer;
    Region new_region;
    for (size_t i = 0; i < region_min.size(); ++i) {
      new_region.push_back(Range::FromMinExtent(region_min[i], region_extents[i]));
    }
    return BufferRegion(buffer, new_region, new_o2i);
  }

  Array<PrimExpr> RewriteBufferIndices(const Buffer& buffer, Array<PrimExpr> indices,
                                       const BufferAllocInfo& info) {
    indices = info.RemoveTrivialDims(indices);
    ICHECK_EQ(indices.size(), info.region.size());
    for (int i = 0; i < info.region.size(); ++i) {
      indices.Set(i, analyzer_.Simplify(indices[i] - info.region[i]->min));
    }
    return indices;
  }

  /*! \brief Map buffer var to the allocation information about each buffer. */
  std::unordered_map<Var, BufferAllocInfo> buffer_info_;
  /*! \brief A mapping from unmodified old blocks to the modified blocks. */
  BlockMap* block_map_;

  arith::Analyzer analyzer_;
};

Stmt BufferCompactorCompact(
    const PrimFunc& f,
    const std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual>& regions,
    const std::unordered_map<Var, StorageAlignAnnotation>& storage_align, bool remove_trivial_dims,
    BlockMap* block_map) {
  arith::Analyzer analyzer;
  // List all dims of `region` that don't have size=1 (or all dims if `remove_trivial_dims` is
  // false).
  auto ListNonTrivialDims = [&analyzer, remove_trivial_dims](const Region& region) {
    std::vector<int> non_trivial_dims;
    for (size_t i = 0; i < region.size(); ++i) {
      if (analyzer.CanProve(region[i]->extent != 1) || !remove_trivial_dims) {
        non_trivial_dims.push_back(i);
      }
    }
    return non_trivial_dims;
  };
  // collect buffer allocation info for no-alias buffers
  std::unordered_map<Var, BufferAllocInfo> buffer_info;
  for (const auto& kv : regions) {
    const Buffer& buffer = kv.first;
    Region region = kv.second;
    auto non_trivial_dims = ListNonTrivialDims(region);
    auto it = storage_align.find(buffer->data);
    auto align = it == storage_align.end() ? std::nullopt
                                           : std::optional<StorageAlignAnnotation>(it->second);
    buffer_info.emplace(buffer->data, BufferAllocInfo(region, buffer, non_trivial_dims, align));
  }
  BufferCompactor compactor(std::move(buffer_info), block_map);
  Stmt stmt = compactor(f->body);
  return stmt;
}

PrimFunc CompactBufferAllocation(PrimFunc f, bool is_strict, bool remove_trivial_dims,
                                 BlockMap* block_map) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    auto region = BufferAccessRegionCollector::Collect(f, /*collect_inbound=*/is_strict);
    auto storage_align = CollectStorageAlignAnnotation(f->body);
    fptr->body = BufferCompactorCompact(f, region, storage_align, remove_trivial_dims, block_map);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass CompactBufferAllocation(bool is_strict, bool remove_trivial_dims) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return CompactBufferAllocation(std::move(f), is_strict, remove_trivial_dims);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CompactBufferAllocation", {});
}

TVM_REGISTER_GLOBAL("tir.transform.CompactBufferAllocation")
    .set_body_typed(CompactBufferAllocation);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
