#pragma once

#include <Eigen/Core>
#include <memory>

namespace gp {
	class Kernel {
	public:
		virtual double evaluate(const Eigen::VectorXd& a, const Eigen::VectorXd& b) = 0;
	};
	using KernelPtr = std::unique_ptr<Kernel>;
}
