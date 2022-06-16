/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"

#include <GaussianProcess/Error.h>

#include <atomic>
#include <random>

namespace gauss::gp::test {
namespace {
static std::atomic<std::size_t> UNIFORM_SEED = 0;

class UniformSampler {
public:
  UniformSampler() : distribution(-1.0, 1.0) {
    this->generator.seed(static_cast<unsigned int>(UNIFORM_SEED));
    UNIFORM_SEED += 10;
  }

  double operator()() const { return this->distribution(this->generator); };

private:
  mutable std::default_random_engine generator;
  mutable std::uniform_real_distribution<double> distribution;
};
} // namespace

std::vector<Eigen::VectorXd> make_samples(const std::size_t samples_numb,
                                          const double lenght,
                                          const Eigen::Index sample_size) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples_numb);
  UniformSampler sampler;
  for (std::size_t k = 0; k < samples_numb; ++k) {
    auto &sample = result.emplace_back(sample_size);
    sample.setZero();
    for (Eigen::Index p = 0; p < sample.size(); ++p) {
      sample(p) = lenght * sampler();
    }
  }
  return result;
}

bool is_zeros(const Eigen::MatrixXd &subject, const double toll) {
  for (Eigen::Index r = 0; r < subject.rows(); ++r) {
    for (Eigen::Index c = 0; c < subject.cols(); ++c) {
      if (std::abs(subject(r, c)) > toll) {
        return false;
      }
    }
  }
  return true;
}

bool is_equal(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b,
              const double toll) {
  return is_zeros(a - b, toll);
}

bool is_equal_vec(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
                  const double toll) {
  if (a.size() != b.size()) {
    return false;
  }
  for (Eigen::Index k = 0; k < a.size(); ++k) {
    if (std::abs(a(k) - b(k)) > toll) {
      return false;
    }
  }
  return true;
}

bool is_symmetric(const Eigen::MatrixXd &subject, const double toll) {
  return is_equal(subject, subject.transpose(), toll);
}

bool is_inverse(const Eigen::MatrixXd &subject,
                const Eigen::MatrixXd &candidate, const double toll) {
  return is_equal(subject * candidate,
                  Eigen::MatrixXd::Identity(subject.rows(), subject.cols()),
                  toll);
}

GridMultiDimensional::GridMultiDimensional(
    const std::size_t size,
    const std::vector<std::array<double, 2>> &intervals) {
  if (intervals.size() < 2) {
    throw std::runtime_error{"too few intervals"};
  }
  for (const auto &interval : intervals) {
    addAxis(size, interval);
  }
}

Eigen::VectorXd GridMultiDimensional::eval() const {
  Eigen::VectorXd result(axis_ranges.size());
  for (std::size_t k = 0; k < axis_ranges.size(); ++k) {
    result(k) = axis_ranges[k]->eval();
  }
  return result;
}

std::vector<std::size_t> GridMultiDimensional::indices() const {
  std::vector<std::size_t> result;
  for (const auto &range : axis_ranges) {
    result.push_back(range->index());
  }
  return result;
}

GridMultiDimensional &GridMultiDimensional::operator++() {
  for (int k = axis_ranges.size() - 1; k > -1; --k) {
    ++(*axis_ranges[k]);
    if ((*axis_ranges[k])()) {
      break;
    }

    if (k != 0) {
      axis_ranges[k] = std::make_unique<samples::Linspace>(*axis_ranges[k]);
    }
  }
  return *this;
}

Eigen::VectorXd GridMultiDimensional::getDeltas() const {
  Eigen::VectorXd result(axis_ranges.size());
  for (std::size_t k = 0; k < axis_ranges.size(); ++k) {
    result(k) = axis_ranges[k]->getDelta();
  }
  return result;
}

std::vector<std::array<double, 2>>
make_intervals(const Eigen::VectorXd &min_corner,
               const Eigen::VectorXd &max_corner) {
  std::vector<std::array<double, 2>> result;
  for (Eigen::Index k = 0; k < min_corner.size(); ++k) {
    result.emplace_back(std::array<double, 2>{min_corner(k), max_corner(k)});
  }
  return result;
}
} // namespace gauss::gp::test
