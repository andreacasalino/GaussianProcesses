/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/components/GaussianProcessBase.h>

namespace gauss::gp {
void train(GaussianProcessBase &process, const std::size_t iterations);
} // namespace gauss::gp
