/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>
#include <GaussianProcess/Error.h>
#include <GaussianProcess/components/OutputMatrix.h>

namespace gauss::gp {
OutputMatrix::~OutputMatrix() = default;

const Eigen::MatrixXd &OutputMatrix::getOutputMatrix() const {
  return output_matrix->access();
}

void OutputMatrix::updateOutputMatrix() {
  output_matrix->expand(getTrainSet()->GetSamplesInput().GetSamples().size());
}

void OutputMatrix::resetOutputMatrix() {
  output_matrix = std::make_unique<SymmetricMatrixExpandable>(
      [this](const Eigen::Index row, const Eigen::Index col) {
        const auto samples =
            this->getTrainSet()->GetSamplesOutput().GetSamples();
        const auto &y1 = samples[static_cast<std::size_t>(row)];
        const auto &y2 = samples[static_cast<std::size_t>(col)];
        return y1.dot(y2);
      });
}

OutputMatrix::OutputMatrix() { resetOutputMatrix(); }
} // namespace gauss::gp
