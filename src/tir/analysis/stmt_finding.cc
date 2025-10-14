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
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

namespace {

auto _FindEntryFunc(const IRModule& mod) {
  // Priority 1: PrimFunc marked as `tir::attr::kIsEntryFunc`
  int num_prim_func = 0;
  std::optional<std::pair<const PrimFuncNode*, GlobalVar>> main_func, last_func;
  for (const auto& [gv, base_func] : mod->functions) {
    if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
      if (func->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
        return decltype(main_func)({func, gv});
      }
      if (gv->name_hint == "main") {
        main_func = {func, gv};
      }
      last_func = {func, gv};
      ++num_prim_func;
    }
  }
  // Priority 2: PrimFunc whose name is `main`
  if (main_func) {
    return main_func;
  }
  // Priority 3: The only PrimFunc in the IRModule
  if (num_prim_func == 1) {
    return last_func;
  }
  return decltype(main_func)(std::nullopt);
}

}  // namespace

const PrimFuncNode* FindEntryFunc(const IRModule& mod, GlobalVar* result_g_var) {
  auto func_gv = _FindEntryFunc(mod);
  if (!func_gv) {
    return nullptr;
  }
  if (result_g_var) {
    *result_g_var = func_gv->second;
  }
  return func_gv->first;
}

Stmt GetEnclosingLoop(const BlockNode* block, Stmt func_body) {
  struct GetRootSeqStmt : public StmtVisitor {
    void VisitStmt_(const SeqStmtNode* seq) override { result = seq; }
    const SeqStmtNode* result;
  };

  struct BlockFinder : public StmtVisitor {
    explicit BlockFinder(const BlockNode* tgt) : target(tgt) {}

    void VisitStmt_(const BlockNode* block) override {
      if (block == target) {
        found = true;
      }
    }

    const BlockNode* target;
    bool found = false;
  };

  GetRootSeqStmt seq_finder;
  seq_finder(func_body);

  ICHECK(seq_finder.result);

  for (auto stmt : seq_finder.result->seq) {
    if (stmt->IsInstance<ForNode>()) {
      BlockFinder finder(block);
      finder(stmt);
      if (finder.found) {
        return stmt;
      }
    }
  }

  LOG(FATAL) << "Enclosing loop not found for a block " << GetRef<Block>(block);
}

const BlockNode* FindAnchorBlock(const IRModule& mod) {
  struct ReductionBlockCollector : public StmtVisitor {
    void VisitStmt_(const BlockNode* block) override {
      if (block->init) {
        blocks.push_back(block);
      }
      StmtVisitor::VisitStmt(block->body);
    }
    std::vector<const BlockNode*> blocks;
  };

  if (auto prim_func = FindEntryFunc(mod, nullptr)) {
    ReductionBlockCollector collector;
    collector(prim_func->body);

    const auto& candidates = collector.blocks;

    if (candidates.empty()) {
      return nullptr;
    } else if (candidates.size() == 1) {
      return candidates[0];
    }

    double best_flops = -1;
    int best_idx = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      auto loop = GetEnclosingLoop(candidates[i], prim_func->body);
      auto flops = EstimateTIRFlops(loop);
      if (flops > best_flops) {
        best_flops = flops;
        best_idx = i;
      }
    }
    return candidates[best_idx];
  }
  return nullptr;
}

TVM_REGISTER_GLOBAL("tir.analysis.find_anchor_block").set_body_typed([](const IRModule& mod) {
  auto ret = FindAnchorBlock(mod);
  if (ret) {
    return Optional<Block>(GetRef<Block>(ret));
  }
  return Optional<Block>(NullOpt);
});

}  // namespace tir
}  // namespace tvm
