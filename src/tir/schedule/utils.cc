#include <tvm/tir/schedule/utils.h>

#include "../../arith/pattern_match.h"
#include "./analysis.h"

namespace tvm::tir {

Array<For> LoopSRefs2Loops(const Array<StmtSRef>& loop_srefs) {
  Array<For> loops;
  loops.reserve(loop_srefs.size());
  for (StmtSRef loop_sref : loop_srefs) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
    loops.push_back(GetRef<For>(loop));
  }
  return loops;
}

Array<StmtSRef> BlockRVs2StmtSRefs(const Schedule& sch, const Array<BlockRV>& block_rvs) {
  Array<StmtSRef> block_srefs;
  block_srefs.reserve(block_rvs.size());
  for (const BlockRV& block_rv : block_rvs) {
    block_srefs.push_back(sch->GetSRef(block_rv));
  }
  return block_srefs;
}

Stmt RemoveFromSeqStmt(const SeqStmt& seq, const Stmt& to_remove) {
  ICHECK_GT(seq->size(), 1);
  Array<Stmt> new_stmts;
  new_stmts.reserve(seq->size());
  for (const Stmt& stmt : seq->seq) {
    if (to_remove.same_as(stmt)) {
      continue;
    }
    if (const auto* realize = stmt.as<BlockRealizeNode>()) {
      if (to_remove.same_as(realize->block)) {
        continue;
      }
    }
    new_stmts.push_back(stmt);
  }
  return SeqStmt::Flatten(new_stmts);
}

Optional<Var> AnalyzeVarWithShift(const PrimExpr& expr, Optional<IntImm>* constant) {
  if (const auto* var = expr.as<VarNode>()) {
    *constant = NullOpt;
    return GetRef<Var>(var);
  }
  arith::PVar<Var> var;
  arith::PVar<IntImm> shift;
  // match: "var + shift"
  if ((var + shift).Match(expr) || (shift + var).Match(expr)) {
    *constant = shift.Eval();
    return var.Eval();
  }
  // match: "var - shift"
  if ((var - shift).Match(expr)) {
    IntImm result = shift.Eval();
    *constant = IntImm(result->dtype, -result->value);
    return var.Eval();
  }
  return NullOpt;
}

void ReorderAndFuseReductionLoops(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                  tir::LoopRV* fused_reduce_loop, size_t* num_spatial_loops) {
  Array<tir::LoopRV> loops = sch->GetLoops(block_rv);
  Array<tir::StmtSRef> loop_srefs;
  for (const tir::LoopRV& loop_rv : loops) {
    loop_srefs.push_back(sch->GetSRef(loop_rv));
  }

  Array<tir::LoopRV> new_order;
  // Step 1. Add spatial loops.
  *num_spatial_loops = 0;
  for (size_t i = 0; i < loops.size(); ++i) {
    if (GetLoopIterType(loop_srefs[i]) == tir::kDataPar) {
      new_order.push_back(loops[i]);
      (*num_spatial_loops)++;
    }
  }
  // Step 2. Add reduction loops.
  Array<tir::LoopRV> reduction_loops;
  for (size_t i = 0; i < loops.size(); ++i) {
    if (GetLoopIterType(loop_srefs[i]) == tir::kCommReduce) {
      new_order.push_back(loops[i]);
      reduction_loops.push_back(loops[i]);
    }
  }
  // Step 3. Apply reordering if new_order differs from the original order.
  ICHECK_EQ(new_order.size(), loops.size());
  for (size_t i = 0; i < loops.size(); ++i) {
    if (!new_order[i].same_as(loops[i])) {
      sch->Reorder(new_order);
      break;
    }
  }
  // Step 4. Fuse all the reduction loops if there are multiple reduction loops.
  CHECK(!reduction_loops.empty()) << "ValueError: There should be at least one reduction loop";
  if (reduction_loops.size() > 1) {
    *fused_reduce_loop = sch->Fuse(reduction_loops);
  } else {
    *fused_reduce_loop = reduction_loops[0];
  }
}

std::unordered_set<std::string> GetBlockNames(const IRModule& mod) {
  struct BlockNameCollector : public tir::StmtVisitor {
    void VisitStmt_(const tir::BlockNode* block) override {
      block_names.insert(block->name_hint);
      StmtVisitor::VisitStmt(block->body);
    }
    std::unordered_set<std::string> block_names;
  };

  if (auto prim_func = tir::FindEntryFunc(mod, nullptr)) {
    BlockNameCollector collector;
    collector(prim_func->body);
    return collector.block_names;
  }
  return {};
}

bool HasBlock(const Schedule& sch, const std::string& block_name) {
  auto block_names = GetBlockNames(sch->mod());
  return block_names.count(block_name);
}

BlockMap BlockMap::CreateFromStmt(const Stmt& stmt) {
  BlockMap block_map;
  PostOrderVisit(stmt, [&block_map](const ObjectRef& obj) {
    if (auto* block_node = obj.as<BlockNode>()) {
      auto block = GetRef<Block>(block_node);
      block_map.fwd_map_.Set(block, block);
      block_map.bwd_map_.Set(block, block);
    }
  });
  return block_map;
}

void BlockMap::Insert(Block b, Block c) {
  // Given an entry `(b -> c)`, find what maps to `b` (call it `a`), and update the entry to
  // `(a -> c)`. If `a` does not exist, then `b` is a new block, and we do nothing.
  auto it = bwd_map_.find(b);
  if (it != bwd_map_.end()) {
    auto a = (*it).second;
    fwd_map_.Set(a, c);
    bwd_map_.Set(c, a);
  }
}

auto GetAllBlocks(const Stmt& stmt) {
  std::unordered_set<const BlockNode*> stmt_blocks;
  PostOrderVisit(stmt, [&](const ObjectRef& obj) {
    if (const auto* block = obj.as<BlockNode>()) {
      stmt_blocks.insert(block);
    }
  });
  return stmt_blocks;
}

void BlockMap::PruneOverStmt(const Stmt& stmt) {
  auto all_blocks = GetAllBlocks(stmt);
  for (auto& [from_blk, to_blk] : fwd_map_) {
    if (!all_blocks.count(to_blk.get())) {
      // Erasing while iterating is safe on Map because it copy-on-writes.
      fwd_map_.erase(from_blk);
      bwd_map_.erase(to_blk);
    }
  }
}

void BlockMap::ValidateOverStmt(const Stmt& stmt) {
  auto all_blocks = GetAllBlocks(stmt);
  for (auto& [_, block] : fwd_map_) {
    ICHECK(all_blocks.count(block.get()))
        << "Block (name = \"" << block->name_hint << "\", addr = " << block.get()
        << ") exists in the block map, but not in the updated program. Stmt = " << stmt;
  }
}

}  // namespace tvm::tir