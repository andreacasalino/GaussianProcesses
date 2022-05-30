/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <Eigen/Core>
#include <functional>
#include <memory>

namespace gauss::gp {
class SymmetricMatrixExpandable {
public:
  SymmetricMatrixExpandable(
      const std::function<double(const Eigen::Index, const Eigen::Index)>
          &emplacer)
      : emplacer(emplacer) {}

  const Eigen::MatrixXd &access() const;

  void expand(const Eigen::Index new_size);
  void reset() { matrix.reset(); }

private:
  const std::function<double(const Eigen::Index, const Eigen::Index)> emplacer;
  std::unique_ptr<Eigen::MatrixXd> matrix;
};

double trace_product(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
} // namespace gauss::gp
