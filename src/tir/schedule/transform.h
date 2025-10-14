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
#ifndef TVM_TIR_SCHEDULE_TRANSFORM_H_
#define TVM_TIR_SCHEDULE_TRANSFORM_H_

#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <utility>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../ir/functor_common.h"

namespace tvm {
namespace tir {

/******** Annotation ********/

/*!
 * \brief Create a new block with the given annotation added
 * \param block The block with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new block with the given annotation as its last annotation
 */
Block WithAnnotation(const BlockNode* block, const String& attr_key, const ObjectRef& attr_value);

/******** Buffer Related ********/

/*!
 * \brief Create a new buffer by changing the storage scope.
 * \param buffer The given buffer.
 * \param scope The target storage scope.
 * \return The new buffer with target storage scope.
 */
Buffer WithScope(const Buffer& buffer, const String& scope);

/*!
 * \brief Create a new buffer by changint the data type.
 * \param buffer The given buffer.
 * \param scope The target data type.
 * \return The new buffer with target data type.
 */
Buffer WithDType(const Buffer& buffer, const DataType& dtype);

/*!
 * \brief Create buffers from given BufferStores, adding 1 dim of given size to each buffer.
 * \param buf_stores The BufferStores of the original block, where the rfactor buffers will be
 * created from
 * \param factor_axis The `factor_axis` parameter of rfactor
 * \param extent The extent (of the rfactor loop, for example) to insert
 * \return The new created buffer
 */
Array<Buffer> CreateExpandedBuffers(const Array<BufferStore>& buf_stores, int factor_axis,
                                    const PrimExpr& extent, const std::string& suffix = ".rf");

/*!
 * \brief Replaces the buffer within the specific sequence of regions
 * \param regions The regions whose buffers are to be replaced
 * \param source The buffer to be replaced
 * \param target The buffer to be replaced to
 * \return The new sequence of regions after replacement
 */
Array<BufferRegion> ReplaceBuffer(Array<BufferRegion> regions, const Buffer& source,
                                  const Buffer& target);

/*!
 * \brief Replaces the buffer within the specific sequence of regions
 * \param regions The regions whose buffers are to be replaced
 * \param buffer_map The mapping from old buffers to new buffers
 * \return The new sequence of regions after replacement
 */
Array<BufferRegion> ReplaceBuffer(Array<BufferRegion> regions,
                                  const Map<Buffer, Buffer>& buffer_map);

/*!
 * \brief Replaces the buffer within the specific sequence of match_buffers
 * \param match_buffers The match_buffers whose buffers are to be replaced
 * \param source The buffer to be replaced
 * \param target The buffer to be replaced to
 * \return The new sequence of match_buffers after replacement
 */
Array<MatchBufferRegion> ReplaceBuffer(Array<MatchBufferRegion> match_buffers, const Buffer& source,
                                       const Buffer& target);

/*!
 * \brief Replaces the buffer region within the specific sequence of regions
 * \param regions The regions to be replaced
 * \param source_buffer The buffer to whose region is to be replaced
 * \param target The buffer region to be replaced to
 * \return The new sequence of regions after replacement
 */
Array<BufferRegion> ReplaceBufferRegion(Array<BufferRegion> regions, const Buffer& source_buffer,
                                        const BufferRegion& target);

/*!
 * \brief Replaces the buffer region within the specific sequence of match_buffers
 * \param regions The match_buffers to be replaced
 * \param source_buffer The buffer to whose region is to be replaced
 * \param target The buffer region to be replaced to
 * \return The new sequence of match_buffers after replacement
 */
Array<MatchBufferRegion> ReplaceBufferRegion(Array<MatchBufferRegion> match_buffers,
                                             const Buffer& source_buffer,
                                             const BufferRegion& target);

/*!
 * \brief A helper mutator which recursively replaces the old buffer with the new buffer and
 * collects the block sref reuse information for the following replacement.
 *
 * If the buffer to be replaced in used as the source in `match_buffers`, depending the specific
 * use cases, the target buffers in `match_buffers` may also need to be mutated. In this
 * case, this class should be subclassed to explicitly handle `match_buffers`.
 */
class ReplaceBufferMutator : public StmtExprMutator {
 public:
  /*!
   * \brief The constructor
   * \param old_buffer The old buffer
   * \param new_buffer The new buffer
   * \param block_sref_reuse Optional map to record mapping between old and new blocks that reuse
   *        sref.
   */
  ReplaceBufferMutator(const Buffer& old_buffer, Buffer new_buffer,
                       Map<Block, Block>* block_sref_reuse);

  ReplaceBufferMutator(const Map<Buffer, Buffer>& buffer_map, Map<Block, Block>* block_sref_reuse);

 protected:
  using StmtExprMutator::VisitExpr_;
  using StmtExprMutator::VisitStmt_;

  PrimExpr VisitExpr_(const VarNode* var) final;

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    auto it = buffer_var_map_.find(node->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      node.CopyOnWrite()->buffer = it->second;
    }
    return node;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override;

  PrimExpr VisitExpr_(const BufferLoadNode* op) override;

  virtual MatchBufferRegion VisitMatchBufferRegion(const MatchBufferRegion& match_buffer);

  Stmt VisitStmt_(const BlockNode* block) override;

  /*!
   * \brief A mapping which maps old buffer vars to new buffers, including the buffers defined in
   * MatchBufferRegion.
   */
  std::unordered_map<const VarNode*, Buffer> buffer_var_map_;
  /*! \brief The block sref reuse map for the following replacement */
  Map<Block, Block>* block_sref_reuse_;
};

/******** Buffer Load Substitution ********/
PrimExpr ReplaceBufferLoads(PrimExpr expr, std::function<PrimExpr(const BufferLoadNode*)> subst);

Stmt ReplaceBufferLoads(Stmt stmt, std::function<PrimExpr(const BufferLoadNode*)> subst);

/******** IterVar-related Mutations ********/

/*!
 * \brief Mutate and replace the block's BlockRealize.
 * \note This action is not straightforward because ScheduleState only supports Block -> Block
 * replacement. This function visits the AST starting from the parent of `block_sref` and applies
 * `mutator` on the BlockRealize.
 * \note This function does not work on the root block.
 * \param self The schedule state
 * \param block_sref The block sref to the block whose BlockRealize is to be mutated
 * \param mutator The mutator function that takes a BlockRealize and returns a new BlockRealize
 * \sa GetBlockRealize
 */
void MutateAndReplaceBlockRealize(ScheduleState& self, const StmtSRef& block_sref,
                                  std::function<BlockRealize(BlockRealize)> mutator);

/*!
 * \brief Convert IterVars of a BlockRealize so that it has an IterVar that maps exactly to `var`.
 * \details The original BlockRealize may have an IterVar that maps to `expr` which contains `var`.
 * Split this IterVar into two parts: one that maps to `var`, and the other that maps to the rest of
 * `expr`.
 * \note This operation requires that only one IterVar's mapping contains `var`, which should be
 * true if the block has affine binding. This property is not checked. It also assumes that `expr`
 * is linear.
 * \param br The BlockRealize
 * \param var The variable to be split out
 * \param range The range of the new IterVar
 * \return A pair (the new BlockRealize, the new IterVar)
 */
std::pair<BlockRealize, IterVar> SplitVarFromIterVars(BlockRealize br, Var var, Range range);

/******** Block Removal ********/

/*!
 * \brief Construct a new AST, with a specific sref tree leaf removed.
 * The leaf's ancestors who have only a single child will be removed too.
 * \param leaf_block_sref The block/loop sref to the sref tree leaf to be removed
 * \param src_stmt The root of the subtree where the replacement begins
 * \param tgt_stmt The root of the subtree after the replacement
 * \return A boolean indicating if the leaf can be removed successfully
 * \note Read before use:
 * 1) Removal is not conducted beyond scope-level.
 * 2) This method only works properly when the scope root is a stage pipeline.
 *
 * An example of the removal plan, say we are removing the leaf block "B" from the AST.
 *
 *  \code
 *    with block([], "scope_root"):
 *        ...
 *        with block([128, 128], "B") as [vi, vj]:
 *            B[vi, vj] = A[vi, vj] + 1.0
 *        with block([128, 128], "C") as [vi, vj]:
 *            C[vi, vj] = B[vi, vj] * 2.0
 *  \endcode
 *
 * Ths method does not mutate the AST, instead it returns the a `(src_stmt, tgt_stmt)` pair as a
 * plan to substitute certain pieces of the IR.
 *
 * In our example, it returns block "scope_root" as `src_stmt`, and the result `tgt_stmt` is:
 *
 *  \code
 *    with block([], "scope_root"):
 *        ...
 *        with block([128, 128], "C") as [vi, vj]:
 *            C[vi, vj] = B[vi, vj] * 2.0
 *  \endcode
 */
void LeafBlockRemovalPlan(const ScheduleState& self, const StmtSRef& leaf_block_sref,
                          Stmt* src_stmt, Stmt* tgt_stmt);

/*!
 * \brief Tile a subset of loops in the block according to the given tensor intrinsic.
 * \param self The schedule to which tiling is applied
 * \param block_rv The block whose subset of loops will be tiled
 * \param intrin_name The name of a tensor intrinsic, must be registerd via
 * TensorIntrin.register(...) beforehand
 * \param allow_padding Whether to allow padding when tiling
 * \return LoopRV corresponding to the outermost loop of a
 * block tiled according to the given intrin, NullOpt if a valid loop mapping is not found
 */
Optional<tir::LoopRV> TileWithTensorIntrin(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                           const String& intrin_name, bool allow_padding = false);

/******** Block mutation ********/

/*!
 * \brief Simplifier for indices of buffer access and block buffer access regions.
 */
class BlockBufferAccessSimplifier : public arith::IRMutatorWithAnalyzer {
 public:
  /*!
   * \brief Simplify indices of buffer access and block buffer access regions in the statement
   * \param stmt The statement to be simplified
   * \param analyzer The arithmetic analyzer
   * \return The simplified statement
   */
  static Stmt Simplify(const Stmt& stmt, arith::Analyzer* analyzer) {
    BlockBufferAccessSimplifier simplifier(analyzer);
    return simplifier(stmt);
  }

 private:
  explicit BlockBufferAccessSimplifier(arith::Analyzer* analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;

  void SimplifyAccessRegion(Array<BufferRegion>* old_access_regions);
  void SimplifyBufferIndices(Array<PrimExpr>* indices);

  Stmt VisitStmt_(const BlockNode* op) final;
  Stmt VisitStmt_(const BufferStoreNode* op) final;
  PrimExpr VisitExpr_(const BufferLoadNode* op) final;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRANSFORM_H_
