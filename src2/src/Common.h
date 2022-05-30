/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <Eigen/Core>
#include <functional>

namespace gauss::gp {
class SymmetricMatrixExpandable {
public:
  SymmetricMatrixExpandable(
      const std::function<double(const Eigen::Index, const Eigen::Index)>
          &emplacer)
      : emplacer(emplacer), matrix(0, 0) {}

  const Eigen::MatrixXd &access() const { return matrix; }

  void expand(const Eigen::Index new_size);

private:
  const std::function<double(const Eigen::Index, const Eigen::Index)> emplacer;
  Eigen::MatrixXd matrix;
};

double trace_product(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
} // namespace gauss::gp
