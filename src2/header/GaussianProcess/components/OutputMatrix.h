/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/components/TrainSetAware.h>

namespace gauss::gp {
class OutputMatrix : virtual public TrainSetAware {
protected:
  OutputMatrix() = default;

  const Eigen::MatrixXd &getOutputMatrix() const;

  void updateOutputMatrix();
  void resetOutputMatrix() { output_matrix.reset(); };

private:
  std::unique_ptr<const Eigen::MatrixXd> output_matrix;
};
} // namespace gauss::gp
