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
class MatrixExpandable {
public:
  const Eigen::MatrixXd &access() const;

  void resize(const Eigen::Index new_size);

protected:
  MatrixExpandable();

  virtual Eigen::MatrixXd makeResized(const Eigen::Index new_size) = 0;

private:
  Eigen::Index size;
  mutable Eigen::MatrixXd computed_portion;
};

class SymmetricMatrixExpandable : public MatrixExpandable {
public:
  using Emplacer =
      std::function<double(const Eigen::Index, const Eigen::Index)>;
  SymmetricMatrixExpandable(const Emplacer &emplacer);

protected:
  Eigen::MatrixXd makeResized(const Eigen::Index new_size) override;

private:
  const Emplacer emplacer;
};

double trace_product(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
} // namespace gauss::gp
