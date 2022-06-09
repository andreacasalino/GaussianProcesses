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

static constexpr double TOLL = 1e-4;

bool is_zeros(const Eigen::MatrixXd &subject);

bool is_equal(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);

bool is_equal_vec(const Eigen::VectorXd &a, const Eigen::VectorXd &b);

bool is_symmetric(const Eigen::MatrixXd &subject);

bool is_inverse(const Eigen::MatrixXd &subject,
                const Eigen::MatrixXd &candidate);
} // namespace gauss::gp::test
