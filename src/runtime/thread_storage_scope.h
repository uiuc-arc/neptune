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

/*!
 * \file thread_storage_scope.h
 * \brief Extract launch parameters configuration from TVMArgs.
 */
#ifndef TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
#define TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_

#include <tvm/runtime/metadata.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/thread_storage_scope.h>

#include <string>
#include <vector>

#include "meta_data.h"

namespace tvm {
namespace runtime {

/*! \brief Launch parameters configuration */
class LaunchParamConfig {
 public:
  void Init(size_t base, const std::vector<std::string>& launch_param_tags) {
    base_ = base;
    std::vector<bool> filled(6, false);
    for (size_t i = 0; i < launch_param_tags.size(); ++i) {
      const std::string& tag = launch_param_tags[i];
      if (tag == launch_param::kUseDynamicSharedMemoryTag) {
        ICHECK_EQ(i, launch_param_tags.size() - 1)
            << "kUseDynamicSharedMemoryTag should be the last tag in launch_param_tags.";
        use_dyn_shared_memory_ = true;
      } else {
        ThreadScope ts = ThreadScope::Create(tag);
        arg_index_map_.push_back(ts.rank * 3 + ts.dim_index);
        filled[ts.rank * 3 + ts.dim_index] = true;
      }
    }
    work_dim_ = 1;
    for (int i = 0; i < 3; ++i) {
      if (filled[i] || filled[i + 3]) {
        work_dim_ = i + 1;
      }
    }
  }
  // extract workload from arguments.
  ThreadWorkLoad Extract(TVMArgs x) const {
    ThreadWorkLoad w;
    std::fill(w.work_size, w.work_size + 6, 1);
    for (size_t i = 0; i < arg_index_map_.size(); ++i) {
      // Dynamic shapes can result in 0 dim size. Guard to ensure that the dim size is at least 1.
      size_t size = static_cast<size_t>(x.values[base_ + i].v_int64);
      if (size > 0) {
        w.work_size[arg_index_map_[i]] = size;
      }
    }
    if (use_dyn_shared_memory_) {
      w.dyn_shmem_size = static_cast<size_t>(x.values[base_ + arg_index_map_.size()].v_int64);
    }
    return w;
  }
  // return the work dim
  size_t work_dim() const { return work_dim_; }

 private:
  /*! \brief base axis */
  size_t base_;
  /*! \brief The worker dimension */
  size_t work_dim_;
  /*! \brief The index mapping. */
  std::vector<uint32_t> arg_index_map_;
  /*! \brief Whether or not use dynamic shared memory. */
  bool use_dyn_shared_memory_{false};
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
