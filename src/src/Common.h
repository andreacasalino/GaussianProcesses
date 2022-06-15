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
class ResizableMatrix {
public:
  void resize(const Eigen::Index new_size);

  const Eigen::MatrixXd &access() const;

  Eigen::Index getSize() const { return size; }
  Eigen::Index getComputedSize() const { return computed_portion_size; }

protected:
  ResizableMatrix() = default;

  virtual Eigen::MatrixXd makeResized() const = 0;
  const Eigen::MatrixXd &getComputedPortion() const { return computed_portion; }

private:
  Eigen::Index size = 0;
  mutable Eigen::Index computed_portion_size = 0;
  mutable Eigen::MatrixXd computed_portion = Eigen::MatrixXd{0, 0};
};

class SymmetricResizableMatrix : public ResizableMatrix {
public:
  using Emplacer =
      std::function<double(const Eigen::Index, const Eigen::Index)>;
  SymmetricResizableMatrix(const Emplacer &emplacer);

protected:
  Eigen::MatrixXd makeResized() const final;

private:
  const Emplacer emplacer;
};

double trace_product(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
} // namespace gauss::gp
