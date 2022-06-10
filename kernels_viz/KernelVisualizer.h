/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

#include <nlohmann/json.hpp>
#include <string>

namespace gauss::gp {
nlohmann::json make_kernel_viz_log(const std::size_t samples_numb,
                                   const KernelFunction &kernel,
                                   const std::string &title,
                                   const std::string &tag);
} // namespace gauss::gp
