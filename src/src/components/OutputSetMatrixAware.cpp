/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/components/OutputSetMatrixAware.h>
#include <Common.h>

namespace gauss::gp {
    namespace {
        Eigen::MatrixXd compute_output_matrix_portion(const std::vector<Eigen::VectorXd>& samples, const MatrixIndices& indices) {
            Eigen::MatrixXd  result(indices.end - indices.start, samples.front().size());
            for (std::size_t i = indices.start; i < indices.end; ++i) {
                result.row(i) = samples[i];
            }
            return result;
        }
    }

    void OutputSetMatrixAware::updateSamplesOutputMatrix() {
        const auto& output_samples = samples->GetSamplesOutput().GetSamples();
        if (nullptr == samples_output_matrix) {
            samples_output_matrix = std::make_unique<Eigen::MatrixXd>(compute_output_matrix_portion(output_samples, MatrixIndices{0, output_samples .size()}));
        }
        else {
            auto new_output_matrix = std::make_unique<Eigen::MatrixXd>(output_samples.size(), samples->GetSamplesOutput().GetSamples().size());
            MatrixIndices old_indices = MatrixIndices{ 0, samples_output_matrix->rows() };
            MatrixIndices new_indices = MatrixIndices{ samples_output_matrix->rows(), output_samples.size() };
            set_matrix_portion(*new_output_matrix, *samples_output_matrix, old_indices, MatrixIndices{ 0, samples_output_matrix->cols() });
            set_matrix_portion(*new_output_matrix, compute_output_matrix_portion(output_samples, new_indices)
                                                 , new_indices, MatrixIndices{ 0, samples_output_matrix->cols() });
            samples_output_matrix = std::move(new_output_matrix);
        }
    }
}
