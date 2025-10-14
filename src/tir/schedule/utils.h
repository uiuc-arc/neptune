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
#ifndef TVM_TIR_SCHEDULE_UTILS_H_
#define TVM_TIR_SCHEDULE_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/node/serialization.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/instruction.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/schedule/trace.h>
#include <tvm/tir/schedule/utils.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/utils.h>

#include "../../arith/pattern_match.h"
#include "../../node/attr_registry.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../support/array.h"
#include "../../support/nd_int_set.h"
#include "./analysis.h"
#include "./error.h"
#include "./instruction_traits.h"
#include "./primitive.h"
#include "./transform.h"

#endif  // TVM_TIR_SCHEDULE_UTILS_H_
