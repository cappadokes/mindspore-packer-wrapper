/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <fstream>
#include <memory>
#include "thread_pool.h"

#include "somas_solver_core.h"
#include "somas_solver_pre.h"
#include "convert_utils_base.h"

namespace mindspore::somas {
constexpr auto kSolBytesThreshold = 100 * 1024 * 1024;
constexpr auto kSolNumThresholdMultiThread = 8;

Status SomasSolverPre::CheckTensors(const TensorsDescMap *pTensors, uint32_t index1, uint32_t index2) {
  auto tensors = *pTensors;
  if (tensors[index1] == nullptr) {
    std::cout << "--EXCEPTION-- " << "NULL tensor received in continuous constraint (tensor index " << index1
              << "), there may be kGraphInput or kGraphOutput in the input tensors or output tensors of the "
                 "fused communication op."
              << std::endl;
    return FAILED;
  }
  if (tensors[index2] == nullptr) {
    std::cout << "--EXCEPTION-- " << "NULL tensor received in continuous constraint (tensor index " << index2
              << "), there may be kGraphInput or kGraphOutput in the input tensors or output tensors of the "
                 "fused communication op."
              << std::endl;
    return FAILED;
  }

  if (tensors[index1]->right_) {
    std::cout << "--WARNING-- " << "Warning:tensor " << index1
              << " already has a right tensor (id: " << tensors[index1]->right_->index_ << std::endl;
  }
  if (tensors[index2]->left_) {
    std::cout << "--WARNING-- " << "Warning:tensor " << index2
              << " already has a left tensor (id: " << tensors[index2]->left_->index_ << std::endl;
  }
  return SUCCESS;
}

Status SomasSolverPre::AddContiguousInfoInMap(const vector<vector<size_t>> &continuous_v, TensorsDescMap *pTensors) {
  auto &tensors = *pTensors;
  // creating S Lists
  for (auto &aux : continuous_v) {
    for (size_t i = 0; i < aux.size() - 1; i++) {
      auto index1 = aux[i];
      auto index2 = aux[i + 1];
      if (CheckTensors(pTensors, SizeToUint(index1), SizeToUint(index2)) == FAILED) {
        return FAILED;
      }
      tensors[index1]->right_ = tensors[index2];
      tensors[index2]->left_ = tensors[index1];
    }
  }
  return SUCCESS;
}

Status SomasSolverPre::AddContiguousInfoInMultiMaps(const vector<vector<size_t>> &continuous_v,
                                                    vector<TensorsDescMap> *vecTensorsMap,
                                                    const TensorsDescMap *pTensors) {
  // creating S Lists
  for (auto &aux : continuous_v) {
    for (size_t i = 0; i < aux.size() - 1; i++) {
      auto index1 = aux[i];
      auto index2 = aux[i + 1];
      if (CheckTensors(pTensors, SizeToUint(index1), SizeToUint(index2)) == FAILED) {
        return FAILED;
      }
      for (auto &tensors_sol : *vecTensorsMap) {
        tensors_sol[index1]->right_ = tensors_sol[index2];
        tensors_sol[index2]->left_ = tensors_sol[index1];
      }
    }
  }
  return SUCCESS;
}

vector<TensorsDescMap> SomasSolverPre::CreateTensorsMaps(const TensorsDescMap &tensors, size_t total_sol) {
  vector<TensorsDescMap> vecTensorsMap(total_sol);
  vecTensorsMap[0] = tensors;
  for (auto &pairT : tensors) {
    for (size_t sol = 1; sol < total_sol; sol++) {
      SomasSolverTensorDesc newDesc = *(pairT.second);
      SomasSolverTensorDescPtr newDescPtr = std::make_shared<SomasSolverTensorDesc>(newDesc);
      (void)vecTensorsMap[sol].emplace(pairT.first, newDescPtr);
    }
  }
  return vecTensorsMap;
}

void FindBest(size_t total_sol, const vector<std::shared_ptr<SomasSolverCore>> &solvers, BestInfo *best_info) {
  assert(best_info != nullptr);
  for (size_t sol = 0; sol < total_sol; sol++) {
    auto &solver = solvers[sol];
    assert(solver != nullptr);
    auto &upperbound = solver->GetUpperbound();
    if (upperbound > best_info->worst) {
      best_info->worst = upperbound;
    }
    if (upperbound >= best_info->best) {
      continue;
    }
    if (best_info->best_algo == kManyObjects && solver->algorithm_ == kSingleObject &&
        best_info->best - upperbound <= kSolBytesThreshold) {
      continue;
    }
    best_info->best = upperbound;
    best_info->best_sol = sol;
    best_info->best_algo = solver->algorithm_;
    best_info->best_timing = LongToSize(solver->timing_);
  }
}

[[maybe_unused]] Status SomasSolverPre::Solving(TensorsDescMap *ptensors,
                                                const std::vector<DynamicBitSet> *pConstraints,
                                                const vector<vector<size_t>> &continuous_v, bool bVerifySolution, bool,
                                                SortingType, FittingType, AlgorithmType) {
  Status ret = SUCCESS;
  try {
    TensorsDescMap &tensors = *ptensors;
    constexpr auto numSortingTypes = static_cast<size_t>(kNumSortingTypes);
    constexpr auto numFittingTypes = static_cast<size_t>(kNumFittingTypes);
    constexpr auto numAlgorithmTypes = static_cast<size_t>(kNumAlgorithmTypes);
    constexpr size_t total_sol = numSortingTypes * numFittingTypes * numAlgorithmTypes;
    const double giga = 1024. * 1024. * 1024.;

    vector<std::shared_ptr<SomasSolverCore>> solvers;
    std::vector<common::Task> tasks;
    vector<TensorsDescMap> vecTensorsMap = CreateTensorsMaps(tensors, total_sol);
    if (AddContiguousInfoInMultiMaps(continuous_v, &vecTensorsMap, ptensors) == FAILED) {
      return FAILED;
    }
    auto start = std::chrono::system_clock::now();
    for (size_t algorithm_strategy = 0, sol = 0; algorithm_strategy < numAlgorithmTypes; algorithm_strategy++) {
      for (size_t sort_strategy = 0; sort_strategy < numSortingTypes; sort_strategy++) {
        for (size_t branching_strategy = 0; branching_strategy < numFittingTypes; branching_strategy++) {
          std::shared_ptr<SomasSolverCore> pSolver =
            std::make_shared<SomasSolverCore>(vecTensorsMap[sol], pConstraints, sol);
          pSolver->SetAlgorithmStrategy(AlgorithmType(algorithm_strategy));
          pSolver->SetSortingStrategy(SortingType(sort_strategy));
          pSolver->SetFittingStrategy(FittingType(branching_strategy));
          pSolver->VerifySolution(bVerifySolution);
          auto task = [pSolver]() {
            return pSolver->MemoryAllocationSolver() == SUCCESS ? common::SUCCESS : common::FAIL;
          };
          tasks.emplace_back(task);
          solvers.emplace_back(pSolver);
          sol++;
        }
      }
    }
    common::ThreadPool::GetInstance().SyncRun(tasks);
    BestInfo best_info;
    FindBest(total_sol, solvers, &best_info);
    auto end = std::chrono::system_clock::now();
    size_t total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto &best_solver = solvers[best_info.best_sol];
    for (auto &tensor : tensors) {
      *(tensor.second) = *(vecTensorsMap[best_info.best_sol][tensor.first]);
    }
    assert(best_solver != nullptr);
    max_offset_ = best_solver->GetUpperbound();
    constexpr float kFloatPresent = 100.0;
    std::cout << "--INFO-- " << "SOMAS SOLVER RESUME:" << std::endl;
    std::cout << "--INFO-- " << "Best Solution:[" << 1 + best_info.best_sol << "/" << total_sol << "] " << std::endl;
    std::cout << "--INFO-- " << "Best result:" << best_info.best << " Bytes " << (double)(best_info.best) / (giga)
              << " GB (" << (double)(best_info.best - best_solver->Getlifelongmemory()) / (giga) << " GB + "
              << (double)best_solver->Getlifelongmemory() / (giga) << " GB from lifelong tensors)" << std::endl;
    std::cout << "--INFO-- " << "Best timing:" << best_info.best_timing << " μs" << std::endl;
    std::cout << "--INFO-- " << "Best algorithm: " << algorithmTypeNames[best_solver->algorithm_] << std::endl;
    std::cout << "--INFO-- " << "Best sorting strategy: " << sortingNames[best_solver->sort_strategy_] << std::endl;
    std::cout << "--INFO-- " << "Best offset strategy: " << branchingNames[best_solver->branching_strategy_]
              << std::endl;
    std::cout << "--INFO-- " << "Time elapsed: " << total_time << " μs" << std::endl;
    std::cout << "--INFO-- " << "Spread:"
              << static_cast<double>((double)(best_info.worst - best_info.best) /
                                     static_cast<double>((double)best_info.best * kFloatPresent))
              << " %%" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "--EXCEPTION-- " << "SomasSolver::Solving FAILED: " << e.what() << std::endl;
  }
  return ret;
}
}  // namespace mindspore::somas
