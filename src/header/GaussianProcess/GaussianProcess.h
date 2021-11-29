/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianUtils/GaussianDistribution.h>
#include <GaussianProcess/kernel/KernelFunction.h>
#include <GaussianProcess/GaussianProcessTrainSet.h>

namespace gauss::gp {
	class GaussianProcess {
	public:

		gauss::GaussianDistribution predict(const Eigen::VectorXd& point) const;

	private:
		KernelFunctionPtr kernel;
		GaussianProcessTrainSet samples;
		std::unique_ptr<Eigen::MatrixXd> covariance_samples;
	};


	class GaussianProcessVectorial {
	public:
		std::vector<gauss::GaussianDistribution> predict(const Eigen::VectorXd& point) const;
	};
}
