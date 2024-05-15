#include <iostream>
#include <cassert>
#include <sstream>
#include <string>
#include <fstream>

#include "somas_solver_pre.h"
#include "somas_tensor.h"

bool is_conflict(size_t start1, size_t end1, size_t start2, size_t end2) {
    if (start2 >= end1 or start1 >= end2) return false;
    return true;
}

namespace mindspore::somas {
    std::vector<SomasTensorPtr> TensorsListFromFile(const std::string &filepath) {
        std::vector<SomasTensorPtr> tensors_list;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filepath << std::endl;
            exit(1);
        }

        std::string header;
        getline(file, header);

        std::string line;
        while (getline(file, line)) {
            std::stringstream ss(line);
            std::string value;

            getline(ss, value, ',');
            auto id = static_cast<size_t>(stoi(value));

            getline(ss, value, ',');
            auto lower = static_cast<size_t>(stoi(value));

            getline(ss, value, ',');
            auto upper = static_cast<size_t>(stoi(value));

            getline(ss, value, ',');
            auto size = static_cast<size_t>(stoi(value));

            auto tensor =
                    std::make_shared<SomasTensor>(id, (size_t) 0, (size_t) 0, size, size,
                                                  kLifeLongNone);
            tensor->lifetime_.start_ = lower;
            tensor->lifetime_.end_ = upper;

            tensors_list.push_back(tensor);
        }
        file.close();
        return tensors_list;
    }

    Status Solve(std::vector<SomasTensorPtr> &tensors_list,
                 std::vector<DynamicBitSet> reuse_matrix) {
        auto tensors_num = tensors_list.size();
        for (const auto &tensor: tensors_list) {
            auto ones_num = reuse_matrix[tensor->GetId()].CountOnesNum();
            tensor->num_constraints_ = tensors_num - ones_num;
        }

        TensorsDescMap solver_tensor_desc_map_;
        for (const auto &tensor: tensors_list) {
            assert(tensor != nullptr);
            if (tensor->GetSolverTensorDesc() != nullptr) {
                SomasSolverTensorDescPtr pSolverTensor = tensor->GetSolverTensorDesc();
                (void) solver_tensor_desc_map_.emplace(pSolverTensor->index_, pSolverTensor);
            }
        }

        SomasSolverPrePtr somas_solver_ = std::make_shared<SomasSolverPre>();
        std::vector<std::vector<size_t>> EmptyContiguousConstraintsVec;
        auto status =
                somas_solver_->Solving(&solver_tensor_desc_map_, &reuse_matrix,
                                       EmptyContiguousConstraintsVec, true);

        for (const auto &tensor: tensors_list) {
            assert(tensor != nullptr);
            tensor->SetOffset();
        }
        return status;
    }

    std::vector<DynamicBitSet> ConflictMap(const std::vector<SomasTensorPtr> &tensors_list) {
        std::vector<DynamicBitSet> reuse_matrix;

        size_t count = tensors_list.back()->GetId() + 1;
        for (size_t i = 0; i < count; i++) {
            (void) reuse_matrix.emplace_back(count);
        }

        for (const auto &tensor_i: tensors_list) {
            for (const auto &tensor_j: tensors_list) {
                if (tensor_i->GetId() == tensor_j->GetId()) continue;
                if (is_conflict(tensor_i->lifetime_.start_, tensor_i->lifetime_.end_, tensor_j->lifetime_.start_,
                                tensor_j->lifetime_.end_)) {
                    reuse_matrix[tensor_i->GetId()].SetBitFalse(tensor_j->GetId());
                } else {
                    reuse_matrix[tensor_i->GetId()].SetBitTrue(tensor_j->GetId());
                }
            }
        }

        return reuse_matrix;
    }

    void SaveToCSV(const std::vector<SomasTensorPtr> &tensors_list) {
        const char *path = std::getenv("MINDSPORE_CSV_PATH");
        const char *name = std::getenv("MINDSPORE_TRACE_NAME");

        if (path && name) {
            std::string path_string = std::string(path);
            std::string filename = std::string(name) + "-out.csv";
            std::string new_path =
                    path_string + "/" + "mindspore-csv-out" + "/";
            std::ofstream outfile_out(new_path + filename, std::ios::trunc);

            if (outfile_out.is_open()) {
                outfile_out << "id,lower,upper,size,offset" << std::endl;
                for (const auto &tensor: tensors_list) {
                    outfile_out << tensor->GetId() << "," << tensor->lifetime_.start_ << ","
                                << tensor->lifetime_.end_ << ","
                                << tensor->GetAlignedSize() << "," << tensor->GetOffset() << std::endl;
                }
                outfile_out.close();
            } else {
                std::cout << "Could not open file: " << new_path << filename
                          << std::endl;
            }
        } else {
            std::cout << "One or both environment variables not found." << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <absolute_path_to_csv_file>" << std::endl;
        return 1;
    }
    std::string filepath = argv[1];

    std::vector<mindspore::somas::SomasTensorPtr> tensors_list = mindspore::somas::TensorsListFromFile(filepath);

    std::vector<mindspore::somas::DynamicBitSet> reuse_matrix = mindspore::somas::ConflictMap(tensors_list);

    if (mindspore::somas::SUCCESS != mindspore::somas::Solve(tensors_list, reuse_matrix)) {
        std::cerr << "Some error occurred, see above for details" << std::endl;
    } else {
        std::cerr << "Success! Saving to csv..." << std::endl;
        mindspore::somas::SaveToCSV(tensors_list);
    }
}
