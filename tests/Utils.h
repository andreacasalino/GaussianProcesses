/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

#include "../samples/Ranges.h"

namespace gauss::gp::test {
std::vector<Eigen::VectorXd> make_samples(const std::size_t samples_numb,
                                          const double lenght,
                                          const Eigen::Index sample_size);

static constexpr double DEFAULT_TOLL = 1e-4;

bool is_zeros(const Eigen::MatrixXd &subject, const double toll = DEFAULT_TOLL);

bool is_equal(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b,
              const double toll = DEFAULT_TOLL);

bool is_equal_vec(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
                  const double toll = DEFAULT_TOLL);

bool is_symmetric(const Eigen::MatrixXd &subject,
                  const double toll = DEFAULT_TOLL);

bool is_inverse(const Eigen::MatrixXd &subject,
                const Eigen::MatrixXd &candidate,
                const double toll = DEFAULT_TOLL);

class GridMultiDimensional {
public:
  template <typename... Intervals>
  GridMultiDimensional(const std::size_t size,
                       const std::array<double, 2> &first,
                       const std::array<double, 2> &second,
                       Intervals... intervals) {
    addAxis(size, first, second, std::forward<Intervals>(intervals)...);
  }

  GridMultiDimensional(const std::size_t size,
                       const std::vector<std::array<double, 2>> &intervals);

  bool operator()() const { return (*axis_ranges.front())(); }
  Eigen::VectorXd eval() const;
  std::vector<std::size_t> indices() const;

  std::size_t getSize() const { return axis_ranges.front()->getSize(); }

  GridMultiDimensional &operator++();

  Eigen::VectorXd getDeltas() const;

private:
  void addAxis(const std::size_t size, const std::array<double, 2> &inteval) {

    axis_ranges.emplace_back(
        std::make_unique<samples::Linspace>(inteval[0], inteval[1], size));
  }

  template <typename... Intervals>
  void addAxis(const std::size_t size, const std::array<double, 2> &inteval,
               Intervals... intervals) {
    addAxis(size, inteval);
    addAxis(size, std::forward<Intervals>(intervals)...);
  }

  using LinspacePtr = std::unique_ptr<samples::Linspace>;
  std::vector<LinspacePtr> axis_ranges;
};

std::vector<std::array<double, 2>>
make_intervals(const Eigen::VectorXd &min_corner,
               const Eigen::VectorXd &max_corner);
} // namespace gauss::gp::test
