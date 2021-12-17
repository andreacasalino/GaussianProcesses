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
	class GaussianProcess
        : public GaussianProcessBase {
	public:
        GaussianProcess(KernelFunctionPtr kernel, const std::size_t input_space_size);

        GaussianProcess(KernelFunctionPtr kernel, gauss::gp::TrainSet train_set);

		gauss::GaussianDistribution predict(const Eigen::VectorXd& point) const;
	};
}
