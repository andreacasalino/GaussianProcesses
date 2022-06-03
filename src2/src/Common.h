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
using Emplacer = std::function<double(const Eigen::Index, const Eigen::Index)>;

class SymmetricMatrixExpandable {
public:
  SymmetricMatrixExpandable(const Emplacer &emplacer);

  const Eigen::MatrixXd &access() const { return matrix; }

  void expand(const Eigen::Index new_size);

private:
  const Emplacer emplacer;
  Eigen::MatrixXd matrix;
};

double trace_product(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
} // namespace gauss::gp
