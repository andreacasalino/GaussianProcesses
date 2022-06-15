/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/GaussianProcess.h>

#include "Ranges.h"

#include <nlohmann/json.hpp>

namespace gauss::gp::samples {
void load(nlohmann::json &recipient,
          const std::vector<Eigen::VectorXd> &subject);

void load(nlohmann::json &recipient,
          const std::vector<std::vector<Eigen::Vector2d>> &subject);

void print(const nlohmann::json &subject, const std::string &file_name);
} // namespace gauss::gp::samples
