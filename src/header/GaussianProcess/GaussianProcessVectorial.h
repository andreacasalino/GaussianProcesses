/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/components/GaussianProcessBase.h>
#include <GaussianUtils/GaussianDistribution.h>

namespace gauss::gp {
    class GaussianProcessVectorial
        : public GaussianProcessBase {
    public:
        GaussianProcessVectorial(KernelFunctionPtr kernel, const std::size_t input_space_size, const std::size_t output_space_size);

        std::vector<gauss::GaussianDistribution> predict(const Eigen::VectorXd& point) const;
    };
}
