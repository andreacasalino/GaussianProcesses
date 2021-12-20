/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/components/TrainSetAware.h>

namespace gauss::gp {
class OutputSetMatrixAware : virtual public TrainSetAware {
protected:
  OutputSetMatrixAware() = default;

  const Eigen::MatrixXd &getSamplesOutputMatrix() const;
  void updateSamplesOutputMatrix();
  void resetSamplesOutputMatrix() { samples_output_matrix.reset(); };

private:
  std::unique_ptr<const Eigen::MatrixXd> samples_output_matrix;
};
} // namespace gauss::gp
