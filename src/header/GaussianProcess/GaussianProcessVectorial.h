/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/KernelAware.h>
#include <GaussianUtils/GaussianDistribution.h>

namespace gauss::gp {
	class GaussianProcessVectorial
		: public KernelAware {
	public:
		std::vector<gauss::GaussianDistribution> predict(const Eigen::VectorXd& point) const;
	};
}
