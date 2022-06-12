/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

namespace gauss::gp::test {
std::vector<Eigen::VectorXd> make_samples(const std::size_t samples_numb,
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
} // namespace gauss::gp::test
