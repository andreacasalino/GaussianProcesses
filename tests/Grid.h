/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <Eigen/Dense>

namespace gauss::gp::test {
class EquispacedGrid {
public:
  EquispacedGrid(const Eigen::VectorXd &min_corner,
                 const Eigen::VectorXd &max_corner, const std::size_t size);

  void
  gridFor(const std::function<void(const Eigen::VectorXd &)> &predicate) const;

  const Eigen::VectorXd &getDeltas() const { return deltas; }

  Eigen::VectorXd at(const std::vector<std::size_t> &indices) const;

  std::vector<std::size_t> randomIndices() const;

private:
  void gridFor_(const std::function<void(const Eigen::VectorXd &)> &predicate,
                const std::vector<std::size_t> &cumulated_indices) const;

  const std::size_t size;
  Eigen::VectorXd deltas;
};
} // namespace gauss::gp::test
