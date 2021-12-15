/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>
#include <functional>

namespace gauss::gp {
    using RadialFunction = std::function<double(const double)>;

    class RadialBasisFunction : public KernelFunction {
    public:
        static const RadialFunction EXP_RADIAL_FUNCTION;

        RadialBasisFunction(const RadialFunction& radial_function = EXP_RADIAL_FUNCTION);

        double evaluate(const Eigen::VectorXd& a,
            const Eigen::VectorXd& b) const override;

        std::unique_ptr<KernelFunction> copy() const override;

    private:
        const RadialFunction radial_function;
    };
} // namespace gauss::gp
