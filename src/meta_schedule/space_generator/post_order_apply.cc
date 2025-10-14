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
#include "tvm/meta_schedule/schedule_rule.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Design Space Generator that generates design spaces by applying schedule rules to blocks
 *  in post-DFS order.
 * */
class PostOrderApplyNode : public SpaceGeneratorNode {
 public:
  /*!
   * \brief Optional block names to target. If not specified all blocks will have spaces generated.
   */
  runtime::PackedFunc f_block_filter_ = nullptr;
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The schedule rules. */
  Optional<Array<ScheduleRule>> sch_rules;

  void VisitAttrs(tvm::AttrVisitor* v) {
    SpaceGeneratorNode::VisitAttrs(v);
    v->Visit("sch_rules", &sch_rules);
    // `rand_state_` is not visited
    // `sch_rules_` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    SpaceGeneratorNode::InitializeWithTuneContext(context);
    this->rand_state_ = ForkSeed(&context->rand_state);
    if (context->target.defined() && !sch_rules.defined()) {
      String kind = GetRuleKindFromTarget(context->target.value());
      if (kind == "llvm") {
        sch_rules = ScheduleRule::DefaultLLVM();
      } else if (kind == "cuda") {
        sch_rules = ScheduleRule::DefaultCUDA();
      } else if (kind == "cuda-tensorcore") {
        sch_rules = ScheduleRule::DefaultCUDATensorCore();
      } else if (kind == "hexagon") {
        sch_rules = ScheduleRule::DefaultHexagon();
      } else if (kind == "vnni") {
        sch_rules = ScheduleRule::DefaultX86("vnni");
      } else if (kind == "avx512") {
        sch_rules = ScheduleRule::DefaultX86("avx512");
      } else if (kind == "c") {
        sch_rules = ScheduleRule::DefaultMicro();
      } else if (kind == "asimd") {
        sch_rules = ScheduleRule::DefaultARM("neon");
      } else if (kind == "dotprod") {
        sch_rules = ScheduleRule::DefaultARM("dotprod");
      } else {
        LOG(FATAL) << "Unsupported kind: " << kind;
        throw;
      }
    }
    if (sch_rules.defined()) {
      for (ScheduleRule rule : sch_rules.value()) {
        rule->InitializeWithTuneContext(context);
      }
    }
  }

  Array<tir::Schedule> GenerateDesignSpace(const IRModule& mod) final {
    using ScheduleAndUnvisitedBlocks = std::pair<tir::Schedule, Array<tir::BlockRV>>;
    CHECK(sch_rules.defined()) << "ValueError: `sch_rules` is not set in PostOrderApply";
    tir::Schedule sch = tir::Schedule::Traced(
        /*mod=*/mod,
        /*rand_state=*/ForkSeed(&this->rand_state_),
        /*debug_mode=*/0,
        /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);

    std::vector<ScheduleAndUnvisitedBlocks> stack;
    Array<tir::Schedule> result{sch};
    Array<tir::BlockRV> all_blocks = BlockCollector::Collect(sch, f_block_filter_);

    for (ScheduleRule sch_rule : sch_rules.value()) {
      for (const tir::Schedule& sch : result) {
        stack.emplace_back(sch, all_blocks);
      }
      result.clear();
      while (!stack.empty()) {
        // get the stack.top()
        auto [sch, blocks] = stack.back();
        stack.pop_back();
        // if all blocks are visited
        if (blocks.empty()) {
          result.push_back(sch);
          continue;
        }
        // otherwise, get the last block that is not visited
        tir::BlockRV block_rv = blocks.back();
        blocks.pop_back();
        if (!sch->HasBlock(block_rv)) {
          stack.emplace_back(sch, blocks);
          continue;
        }
        if (!ScheduleRule::IsApplyCustomRule(sch_rule)) {
          if (tir::GetAnn<String>(sch->GetSRef(block_rv), "schedule_rule").defined()) {
            stack.emplace_back(sch, blocks);
            continue;
          }
        }
        Array<tir::Schedule> applied = sch_rule->Apply(sch, /*block=*/block_rv);
        for (const tir::Schedule& sch : applied) {
          stack.emplace_back(sch, blocks);
        }
      }
    }
    return result;
  }

  SpaceGenerator Clone() const final {
    ObjectPtr<PostOrderApplyNode> n = make_object<PostOrderApplyNode>(*this);
    CloneRules(this, n.get());
    if (this->sch_rules.defined()) {
      Array<ScheduleRule> original = this->sch_rules.value();
      Array<ScheduleRule> sch_rules;
      sch_rules.reserve(original.size());
      for (const ScheduleRule& sch_rule : original) {
        sch_rules.push_back(sch_rule->Clone());
      }
      n->sch_rules = std::move(sch_rules);
    }
    return SpaceGenerator(n);
  }
  static constexpr const char* _type_key = "meta_schedule.PostOrderApply";
  TVM_DECLARE_FINAL_OBJECT_INFO(PostOrderApplyNode, SpaceGeneratorNode);
};

SpaceGenerator SpaceGenerator::PostOrderApply(runtime::PackedFunc f_block_filter,
                                              Optional<Array<ScheduleRule>> sch_rules,
                                              Optional<Array<Postproc>> postprocs,
                                              Optional<Map<Mutator, FloatImm>> mutator_probs) {
  ObjectPtr<PostOrderApplyNode> n = make_object<PostOrderApplyNode>();
  n->sch_rules = std::move(sch_rules);
  n->postprocs = std::move(postprocs);
  n->mutator_probs = std::move(mutator_probs);
  n->f_block_filter_ = std::move(f_block_filter);
  return SpaceGenerator(n);
}

TVM_REGISTER_NODE_TYPE(PostOrderApplyNode);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorPostOrderApply")
    .set_body_typed(SpaceGenerator::PostOrderApply);

}  // namespace meta_schedule
}  // namespace tvm
