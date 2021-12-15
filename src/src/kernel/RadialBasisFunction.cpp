/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/RadialBasisFunction.h>

namespace gauss::gp {
    RadialBasisFunction::RadialBasisFunction(const RadialFunction& radial_function = EXP_RADIAL_FUNCTION) 
        : radial_function(radial_function) {};

    double RadialBasisFunction::evaluate(const Eigen::VectorXd& a,
        const Eigen::VectorXd& b) const {
        auto diff = a - b;
        return radial_function(diff.squaredNorm());
    };

    std::unique_ptr<KernelFunction> RadialBasisFunction::copy() const {
        return std::make_unique<RadialBasisFunction>(radial_function);
    };

    const RadialFunction RadialBasisFunction::EXP_RADIAL_FUNCTION = [](const double dist) { return exp(dist); };
}
