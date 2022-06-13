/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/GaussianProcess.h>

#include <array>
#include <functional>
#include <nlohmann/json.hpp>

namespace gauss::gp::samples {
std::vector<Eigen::VectorXd>
make_equispaced_input_samples(const double min, const double max,
                              const std::size_t size);

std::vector<std::vector<Eigen::VectorXd>>
make_equispaced_input_samples(const std::array<double, 2> &min,
                              const std::array<double, 2> &max,
                              const std::size_t size);

using Function = std::function<double(const Eigen::VectorXd &)>;

void convert(nlohmann::json &recipient,
             const std::vector<Eigen::VectorXd> &subject);

void print(const nlohmann::json &subject, const std::string &file_name);
} // namespace gauss::gp::samples
