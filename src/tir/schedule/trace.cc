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
#include <tvm/tir/sexpr_printer.h>

#include "./utils.h"

namespace tvm {
namespace tir {

/**************** Constructors  ****************/

Trace::Trace() { data_ = make_object<TraceNode>(); }

Trace::Trace(Array<Instruction> insts, Map<Instruction, ObjectRef> decisions) {
  ObjectPtr<TraceNode> n = make_object<TraceNode>();
  n->insts = std::move(insts);
  n->decisions = std::move(decisions);
  data_ = std::move(n);
}

/**************** Utilities  ****************/

int GetNumValidInstructions(const Array<Instruction>& insts, bool remove_postproc) {
  if (!remove_postproc) {
    return insts.size();
  }
  int n_insts = 0;
  for (const Instruction& inst : insts) {
    if (!inst->kind->IsPostproc()) {
      ++n_insts;
    } else {
      break;
    }
  }
  return n_insts;
}

/**************** TranslateInputRVs  ****************/

Array<ObjectRef> TranslateInputRVs(const Array<ObjectRef>& inputs,
                                   const std::unordered_map<const Object*, const Object*>& rv_map) {
  Array<ObjectRef> result;
  result.reserve(inputs.size());
  auto f_subst_with_rv_map = [&rv_map](const Var& var) -> Optional<PrimExpr> {
    auto it = rv_map.find(var.get());
    if (it == rv_map.end()) {
      return NullOpt;
    }
    const Object* dst = it->second;
    ICHECK(dst->IsInstance<VarNode>())
        << "TypeError: Expect 'tir.Var', but gets: " << dst->GetTypeKey();
    return GetRef<Var>(static_cast<const VarNode*>(dst));
  };

  for (const ObjectRef& input : inputs) {
    if (!input.defined() || input->IsInstance<StringObj>() || input->IsInstance<IntImmNode>() ||
        input->IsInstance<FloatImmNode>() || input->IsInstance<runtime::Int::ContainerType>() ||
        input->IsInstance<runtime::Float::ContainerType>()) {
      result.push_back(input);
    } else if (input->IsInstance<BlockRVNode>() ||  // RV: block
               input->IsInstance<LoopRVNode>() ||   // RV: loop
               input->IsInstance<VarNode>()) {      // RV: var
      auto it = rv_map.find(input.get());
      ICHECK(it != rv_map.end()) << "IndexError: Random variable doesn't exist: " << input;
      result.push_back(GetRef<ObjectRef>(it->second));
    } else if (auto expr = input.as<PrimExpr>()) {  // RV: Expr
      result.push_back(Substitute(expr.value(), f_subst_with_rv_map));
    } else if (auto index_map = input.as<IndexMap>()) {
      result.push_back(Substitute(index_map.value(), f_subst_with_rv_map));
    } else if (auto arr = input.as<Array<ObjectRef>>()) {
      // Recursively convert elements of the array into a new list of ObjectRefs.
      result.push_back(TranslateInputRVs(arr.value(), rv_map));
    } else {
      ICHECK(false) << "TypeError: Cannot recognize the type of an input random variable: "
                    << input->GetTypeKey();
      throw;
    }
  }
  return result;
}

void TranslateAddOutputRVs(const Array<ObjectRef>& old_outputs, const Array<ObjectRef>& new_outputs,
                           std::unordered_map<const Object*, const Object*>* rv_map) {
  ICHECK_EQ(old_outputs.size(), new_outputs.size());
  int n = old_outputs.size();
  const ObjectRef* p_old = old_outputs.GetArrayNode()->begin();
  const ObjectRef* p_new = new_outputs.GetArrayNode()->begin();
  for (int i = 0; i < n; ++i) {
    (*rv_map)[p_old[i].get()] = p_new[i].get();
  }
}

enum class EscapeMode {
  kPython,
  kJSON,
};

// More specific version of TranslateInputRVs that specifically maps from in-TVM repr
// to outside repr.
Array<ObjectRef> InputRVsToNames(const Array<ObjectRef>& inputs,
                                 const std::unordered_map<const Object*, String>& rv_names,
                                 EscapeMode mode) {
  Array<ObjectRef> results;
  results.reserve(inputs.size());
  for (const ObjectRef& input : inputs) {
    if (!input.defined()) {
      // Case 0. nullptr => None
      switch (mode) {
        case EscapeMode::kPython:
          results.push_back(String("None"));
          break;
        case EscapeMode::kJSON:
          results.push_back(ObjectRef());
          break;
      }
      continue;
    }
    if (input->IsInstance<BlockRVNode>() || input->IsInstance<LoopRVNode>() ||
        input->IsInstance<VarNode>()) {
      // Case 1. BlockRV, LoopRV, VarRV
      auto it = rv_names.find(input.get());
      ICHECK(it != rv_names.end()) << "IndexError: Random variable is not defined " << input;
      results.push_back(it->second);
    } else if (const auto* str_obj = input.as<StringObj>()) {
      // Case 2. string => "content"
      results.push_back(String('"' + std::string(str_obj->data) + '"'));
    } else if (input->IsInstance<IntImmNode>() || input->IsInstance<FloatImmNode>() ||
               input->IsInstance<runtime::Int::ContainerType>() ||
               input->IsInstance<runtime::Float::ContainerType>()) {
      // Case 3. integer or floating-point number
      results.push_back(input);
    } else if (input->IsInstance<ArrayNode>()) {
      // Case 4: array
      results.push_back(InputRVsToNames(Downcast<Array<ObjectRef>>(input), rv_names, mode));
    } else if (input->IsInstance<MapNode>()) {
      // Case 5: dict
      results.push_back(input);
    } else if (auto index_map_ = input.as<IndexMap>()) {
      // Case 6: IndexMap
      IndexMap index_map =
          index_map_.value().RenameVariables([&rv_names](const Var& var) -> Optional<String> {
            if (auto it = rv_names.find(var.get()); it != rv_names.end()) {
              return it->second;
            }
            return NullOpt;
          });
      switch (mode) {
        // Send the IndexMap directly. Ultimately `UnpackedAsPython()` gets it in arguments,
        // and will handle it properly.
        case EscapeMode::kPython:
          results.push_back(index_map);
          break;
        // Convert the IndexMap to JSON. This is a bit tricky as TVM's JSONify mechanism
        // returns a string of a JSON object, while ideally we want a dict.
        // Instead we'll just play along with this string, pretending it's a dict.
        // (Hey, it's shallow embedding!)
        case EscapeMode::kJSON:
          results.push_back(String(SaveJSON(index_map)));
          break;
      }
    } else if (auto expr_ = input.as<PrimExpr>()) {
      // We can have expressions of VarRVs in the trace, which is handled similarly to the IndexMap
      // case above.
      PrimExpr expr = Substitute(expr_.value(), [&rv_names](const Var& var) -> Optional<PrimExpr> {
        if (auto it = rv_names.find(var.get()); it != rv_names.end()) {
          return Var(it->second);
        }
        return NullOpt;
      });
      switch (mode) {
        case EscapeMode::kPython:
          results.push_back(expr);
          break;
        // Convert the expr to an S-expr string. This is much (much!) shorter than what SaveJSON
        // will do, and the result is somewhat readable (compared to the garbled mess of SaveJSON).
        case EscapeMode::kJSON:
          results.push_back(String(SExprPrinter::Print(expr)));
          break;
      }
    } else {
      LOG(FATAL) << "TypeError: Stringifying is not supported for type: " << input->GetTypeKey();
      throw;
    }
  }
  return results;
}

// A more specific version of TranslateInputRVs that specifically maps from JSON repr
// to in-TVM repr. Python repr -> TVM is not handled here (handled by the interpreter instead, in
// effect).
Array<ObjectRef> InputNamesToRVs(const Array<ObjectRef>& inputs,
                                 const std::unordered_map<std::string, ObjectRef>& named_rvs) {
  Array<ObjectRef> results;
  results.reserve(inputs.size());
  for (const ObjectRef& input : inputs) {
    if (!input.defined()) {
      // Case 0. None
      results.push_back(ObjectRef());
    } else if (input->IsInstance<IntImmNode>() || input->IsInstance<FloatImmNode>() ||
               input->IsInstance<runtime::Int::ContainerType>() ||
               input->IsInstance<runtime::Float::ContainerType>()) {
      // Case 3. integer or floating-point number
      results.push_back(input);
    } else if (input->IsInstance<ArrayNode>()) {
      // Case 4. array
      results.push_back(InputNamesToRVs(Downcast<Array<ObjectRef>>(input), named_rvs));
    } else if (input->IsInstance<MapNode>()) {
      // Case 5. dict
      results.push_back(input);
    } else {
      // Everything down here is a JSON string, but a JSON string can mean multiple TVM things.
      // Cases (note that they don't overlap):
      //   2. string -- a quoted string "..."
      //   6. IndexMap -- an object { ... }
      //   1. PrimExpr, BlockRV, LoopRV, VarRV -- anything that can be parsed into a PrimExpr.
      //      A legal repr of PrimExpr is never surrounded by "" or {}.
      const auto* str = input.as<StringObj>();
      CHECK(str) << "TypeError: Expect String, but gets: " << input->GetTypeKey();
      std::string name = GetRef<String>(str);
      int64_t size = name.size();
      CHECK_GT(size, 0) << "ValueError: Empty string is not allowed in input names";
      if (name.front() == '{' && name.back() == '}') {
        // Case 6. IndexMap
        IndexMap index_map = Downcast<IndexMap>(LoadJSON(name));
        index_map = Substitute(index_map, [&named_rvs](const Var& var) -> Optional<PrimExpr> {
          auto it = named_rvs.find(var->name_hint);
          if (it != named_rvs.end()) {
            return Downcast<Var>(it->second);
          }
          return NullOpt;
        });
        results.push_back(index_map);
      } else if (name.front() == '"' && name.back() == '"') {
        // Case 2. string
        ICHECK(size >= 2);
        results.push_back(String(name.substr(1, size - 2)));
      } else {
        auto it = named_rvs.find(name);
        if (it != named_rvs.end()) {
          // Case 1. BlockRV, LoopRV, VarRV
          results.push_back(it->second);
        } else {
          // Case 7. PrimExpr
          results.push_back(SExprParser::Parse(name, [&named_rvs](const std::string& name) {
            auto it = named_rvs.find(name);
            ICHECK(it != named_rvs.end()) << "IndexError: Random variable doesn't exist: " << name;
            return Downcast<Var>(it->second);
          }));
        }
      }
    }
  }
  return results;
}

Array<String> OutputRVsToNames(const Array<ObjectRef>& outputs,
                               std::unordered_map<const Object*, String>& rv_names) {
  Array<String> results;
  results.reserve(outputs.size());
  for (const ObjectRef& output : outputs) {
    int i = rv_names.size();
    auto it = rv_names.find(output.get());
    ICHECK(it == rv_names.end()) << "ValueError: The random variable has been produced once: "
                                 << it->second;
    String result{nullptr};
    if (output->IsInstance<BlockRVNode>()) {
      result = "b" + std::to_string(i);
    } else if (output->IsInstance<LoopRVNode>()) {
      result = "l" + std::to_string(i);
    } else if (auto var_node = output.as<VarNode>()) {
      result = var_node->name_hint;
    } else {
      LOG(FATAL) << "TypeError: Cannot recognize the type of the random variable: "
                 << output->GetTypeKey();
      throw;
    }
    results.push_back(result);
    rv_names.emplace(output.get(), std::move(result));
  }
  return results;
}

void UpdateOutputNamesToRVs(const Array<String>& old_outputs, const Array<ObjectRef>& new_outputs,
                            std::unordered_map<std::string, ObjectRef>& named_rvs) {
  ICHECK_EQ(old_outputs.size(), new_outputs.size());
  int n = old_outputs.size();
  const ObjectRef* p_old = old_outputs.GetArrayNode()->begin();
  const ObjectRef* p_new = new_outputs.GetArrayNode()->begin();
  for (int i = 0; i < n; ++i) {
    const auto* name = static_cast<const StringObj*>(p_old[i].get());
    named_rvs.emplace(std::string(name->data, name->size), p_new[i]);
  }
}

/**************** Add/Remove/Get ****************/

Optional<ObjectRef> TraceNode::GetDecision(const Instruction& inst) const {
  auto it = this->decisions.find(inst);
  return it == this->decisions.end() ? Optional<ObjectRef>(NullOpt) : (*it).second;
}

void TraceNode::Append(Instruction inst) { insts.push_back(std::move(inst)); }

void TraceNode::Append(Instruction inst, ObjectRef decision) {
  decisions.Set(inst, std::move(decision));
  insts.push_back(std::move(inst));
}

Optional<Instruction> TraceNode::Pop() {
  if (insts.empty()) {
    return NullOpt;
  }
  Instruction inst = insts.back();
  insts.pop_back();
  if (decisions.count(inst)) {
    decisions.erase(inst);
  }
  return inst;
}

/**************** Interfacing with InstructionKind ****************/

void TraceNode::ApplyToSchedule(
    Schedule sch, bool remove_postproc,
    runtime::TypedPackedFunc<ObjectRef(const Instruction& inst, const Array<ObjectRef>& inputs,  //
                                       const Array<ObjectRef>& attrs,                            //
                                       const Optional<ObjectRef>& decision)>
        decision_provider) const {
  std::unordered_map<const Object*, const Object*> rv_map;
  for (const Instruction& inst : this->insts) {
    if (remove_postproc && inst->kind->IsPostproc()) {
      break;
    }
    Array<ObjectRef> inputs = TranslateInputRVs(inst->inputs, rv_map);
    Array<ObjectRef> attrs = inst->attrs;
    Optional<ObjectRef> decision = this->GetDecision(inst);
    if (decision_provider != nullptr) {
      decision = decision_provider(inst, inputs, attrs, decision);
    }
    Array<ObjectRef> outputs = inst->kind->f_apply_to_schedule(sch, inputs, attrs, decision);
    TranslateAddOutputRVs(inst->outputs, outputs, &rv_map);
  }
}

ObjectRef TraceNode::AsJSON(bool remove_postproc) const {
  std::unordered_map<const Object*, String> rv_names;
  Array<ObjectRef> json_insts;
  Array<ObjectRef> json_decisions;
  json_insts.reserve(this->insts.size());
  json_decisions.reserve(this->insts.size());

  int i = 0;
  for (const Instruction& inst : this->insts) {
    const InstructionKind& kind = inst->kind;
    if (remove_postproc && kind->IsPostproc()) {
      break;
    }
    json_insts.push_back(Array<ObjectRef>{
        /* 0: inst name */ kind->name,
        /* 1: inputs    */ InputRVsToNames(inst->inputs, rv_names, EscapeMode::kJSON),
        /* 2: attrs     */ kind->f_attrs_as_json != nullptr ? kind->f_attrs_as_json(inst->attrs)
                                                            : ObjectRef(inst->attrs),
        /* 3: outputs   */ OutputRVsToNames(inst->outputs, rv_names),
    });
    if (Optional<ObjectRef> decision = this->GetDecision(inst)) {
      json_decisions.push_back(Array<ObjectRef>{
          /* 0: index    */ Integer(i),
          /* 1: decision */ decision.value(),
      });
    }
    ++i;
  }
  return Array<ObjectRef>{
      /* 0: trace    */ std::move(json_insts),
      /* 1: decision */ std::move(json_decisions),
  };
}

Array<String> TraceNode::AsPython(bool remove_postproc) const {
  std::unordered_map<const Object*, String> rv_names;
  Array<String> py_trace;
  py_trace.reserve(this->insts.size());
  for (const Instruction& inst : this->insts) {
    if (remove_postproc && inst->kind->IsPostproc()) {
      break;
    }
    Array<ObjectRef> attrs;
    attrs.reserve(inst->attrs.size());
    for (const ObjectRef& obj : inst->attrs) {
      if (const auto* str = obj.as<StringObj>()) {
        attrs.push_back(String('"' + std::string(str->data) + '"'));
      } else {
        attrs.push_back(obj);
      }
    }
    py_trace.push_back(inst->kind->f_as_python(
        /*inputs=*/InputRVsToNames(inst->inputs, rv_names, EscapeMode::kPython),
        /*attrs=*/attrs,
        /*decision=*/this->GetDecision(inst),
        /*outputs=*/OutputRVsToNames(inst->outputs, rv_names)));
  }
  return py_trace;
}

void Trace::ApplyJSONToSchedule(ObjectRef json, Schedule sch) {
  Array<ObjectRef> json_insts{nullptr};
  Array<ObjectRef> json_decisions{nullptr};
  // Parse `json` into `json_insts` and `json_decisions`
  try {
    const ArrayNode* arr = json.as<ArrayNode>();
    ICHECK(arr && arr->size() == 2);
    const auto* arr0 = arr->at(0).as<ArrayNode>();
    const auto* arr1 = arr->at(1).as<ArrayNode>();
    ICHECK(arr0 && arr1);
    json_insts = GetRef<Array<ObjectRef>>(arr0);
    json_decisions = GetRef<Array<ObjectRef>>(arr1);
  } catch (const tvm::Error& e) {
    LOG(FATAL) << "ValueError: The json entry of a trace should contain two arrays, an array of "
                  "instructions and an array of decisions, but gets: "
               << json;
    throw;
  }
  // Parse `json_decisions`
  std::vector<Optional<ObjectRef>> decisions(json_insts.size(), NullOpt);
  for (const ObjectRef& decision_entry : json_decisions) {
    int index = -1;
    ObjectRef decision{nullptr};
    try {
      const ArrayNode* arr = decision_entry.as<ArrayNode>();
      ICHECK(arr && arr->size() == 2);
      auto arr0 = arr->at(0).as<runtime::Int>();
      ICHECK(arr0);
      index = arr0.value();
      decision = arr->at(1);
    } catch (const tvm::Error& e) {
      LOG(FATAL) << "ValueError: Each entry of a json decision should be a tuple [index, "
                    "decision], but gets: "
                 << decision_entry;
      throw;
    }
    decisions[index] = std::move(decision);
  }
  // Parse `json_insts`
  std::unordered_map<std::string, ObjectRef> named_rvs{{"None", ObjectRef{nullptr}}};
  int i = 0;
  for (const ObjectRef& inst_entry : json_insts) {
    InstructionKind kind{nullptr};
    Array<ObjectRef> inputs{nullptr};
    Array<ObjectRef> attrs{nullptr};
    Array<String> outputs{ObjectPtr<Object>{nullptr}};
    // Parse the entry
    try {
      const auto* arr = inst_entry.as<ArrayNode>();
      ICHECK(arr && arr->size() == 4);
      const auto* arr0 = arr->at(0).as<StringObj>();
      const auto* arr1 = arr->at(1).as<ArrayNode>();
      const auto* arr2 = arr->at(2).as<ArrayNode>();
      const auto* arr3 = arr->at(3).as<ArrayNode>();
      ICHECK(arr0 && arr1 && arr2 && arr3);
      for (const ObjectRef& str : *arr3) {
        ICHECK(str->IsInstance<StringObj>());
      }
      kind = InstructionKind::Get(arr0->data);
      inputs = GetRef<Array<ObjectRef>>(arr1);
      attrs = GetRef<Array<ObjectRef>>(arr2);
      outputs = GetRef<Array<String>>(arr3);
    } catch (const tvm::Error& e) {
      LOG(FATAL) << "ValueError: Each entry of a json instruction should be a tuple [inst_name, "
                    "inputs, attrs, outputs], but gets: "
                 << inst_entry;
      throw;
    }
    // Parse inputs
    inputs = InputNamesToRVs(inputs, named_rvs);
    // Parse attrs
    if (kind->f_attrs_from_json != nullptr) {
      attrs = kind->f_attrs_from_json(attrs);
    }
    // Apply to the schedule
    Array<ObjectRef> new_outputs = kind->f_apply_to_schedule(sch, inputs, attrs, decisions[i]);
    // Parse outputs
    UpdateOutputNamesToRVs(outputs, new_outputs, named_rvs);
    ++i;
  }
}

/**************** Creation ****************/

Trace TraceNode::WithDecision(Instruction inst, ObjectRef decision, bool remove_postproc) const {
  int n_insts = GetNumValidInstructions(this->insts, remove_postproc);
  Array<Instruction> new_insts =
      Array<Instruction>{this->insts.begin(), this->insts.begin() + n_insts};
  Map<Instruction, ObjectRef> new_decisions{this->decisions.begin(), this->decisions.end()};
  new_decisions.Set(std::move(inst), std::move(decision));
  return Trace(new_insts, new_decisions);
}

Trace TraceNode::Simplified(bool remove_postproc) const {
  int n_insts = GetNumValidInstructions(this->insts, remove_postproc);
  std::unordered_set<const Object*> used_rvs;
  std::vector<Instruction> new_insts;
  std::unordered_map<Instruction, ObjectRef, ObjectPtrHash, ObjectPtrEqual> new_decisions;
  new_insts.reserve(n_insts);
  new_decisions.reserve(this->decisions.size());
  for (int inst_idx = n_insts - 1; inst_idx >= 0; --inst_idx) {
    const Instruction& inst = this->insts[inst_idx];
    // Check if all the variables the instruction defined are dead
    // If so, and the instruction is pure, we can safely remove this instruction
    bool all_defs_dead = inst->kind->is_pure;
    if (all_defs_dead) {
      for (const ObjectRef& obj : inst->outputs) {
        if (used_rvs.count(obj.get())) {
          all_defs_dead = false;
          break;
        }
      }
    }
    // Remove this instruction
    if (all_defs_dead) {
      continue;
    }
    // Otherwise this instruction is not dead
    new_insts.push_back(inst);
    if (Optional<ObjectRef> decision = this->GetDecision(inst)) {
      new_decisions.emplace(inst, std::move(decision));
    }
    // Add its inputs as "used" ones
    for (const ObjectRef& obj : inst->inputs) {
      if (!obj.defined()) {
        continue;
      } else if (obj->IsInstance<BlockRVNode>() || obj->IsInstance<LoopRVNode>() ||
                 obj->IsInstance<VarNode>()) {
        used_rvs.insert(obj.get());
        continue;
      } else if (obj->IsInstance<PrimExprNode>()) {
        PostOrderVisit(obj, [&used_rvs](const ObjectRef& obj) -> void {
          if (obj->IsInstance<VarNode>()) {
            used_rvs.insert(obj.get());
          }
        });
      }
    }
  }
  return Trace(Array<Instruction>(new_insts.rbegin(), new_insts.rend()),
               Map<Instruction, ObjectRef>(new_decisions));
}

/**************** Repr ****************/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TraceNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* self = obj.as<TraceNode>();
      ICHECK_NOTNULL(self);
      p->stream << "# from tvm import tir\n";
      p->stream << "def apply_trace(sch: tir.Schedule) -> None:\n";
      Array<String> repr = self->AsPython(/*remove_postproc=*/false);
      bool is_first = true;
      for (const String& line : repr) {
        if (is_first) {
          is_first = false;
        } else {
          p->stream << '\n';
        }
        p->stream << "  " << line;
      }
      if (is_first) {
        p->stream << "  pass";
      }
      p->stream << std::flush;
    });

/**************** Instruction Registration ****************/

struct EnterPostprocTraits : public UnpackedInstTraits<EnterPostprocTraits> {
  static constexpr const char* kName = "EnterPostproc";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch) { return sch->EnterPostproc(); }

  static String UnpackedAsPython(Array<String> outputs) {
    PythonAPICall py("enter_postproc");
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(EnterPostprocTraits);

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(TraceNode);
TVM_REGISTER_GLOBAL("tir.schedule.Trace")
    .set_body_typed([](Optional<Array<Instruction>> insts,
                       Optional<Map<Instruction, ObjectRef>> decisions) {
      return Trace(insts.value_or(Array<Instruction>()),
                   decisions.value_or(Map<Instruction, ObjectRef>()));
    });
TVM_REGISTER_GLOBAL("tir.schedule.TraceGetDecision")
    .set_body_method<Trace>(&TraceNode::GetDecision);
TVM_REGISTER_GLOBAL("tir.schedule.TraceAppend")
    .set_body_typed([](Trace self, Instruction inst, Optional<ObjectRef> decision) {
      if (decision.defined()) {
        return self->Append(inst, decision.value());
      } else {
        return self->Append(inst);
      }
    });
TVM_REGISTER_GLOBAL("tir.schedule.TracePop").set_body_method<Trace>(&TraceNode::Pop);
TVM_REGISTER_GLOBAL("tir.schedule.TraceApplyToSchedule")
    .set_body_method<Trace>(&TraceNode::ApplyToSchedule);
TVM_REGISTER_GLOBAL("tir.schedule.TraceAsJSON").set_body_method<Trace>(&TraceNode::AsJSON);
TVM_REGISTER_GLOBAL("tir.schedule.TraceAsPython").set_body_method<Trace>(&TraceNode::AsPython);
TVM_REGISTER_GLOBAL("tir.schedule.TraceWithDecision")
    .set_body_method<Trace>(&TraceNode::WithDecision);
TVM_REGISTER_GLOBAL("tir.schedule.TraceSimplified").set_body_method<Trace>(&TraceNode::Simplified);
TVM_REGISTER_GLOBAL("tir.schedule.TraceApplyJSONToSchedule")
    .set_body_typed(Trace::ApplyJSONToSchedule);

}  // namespace tir
}  // namespace tvm
