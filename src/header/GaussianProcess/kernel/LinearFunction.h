/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

namespace gauss::gp {
    class LinearFunction : public KernelFunction {
    public:
        LinearFunction(const double teta0, const double teta1);

        double evaluate(const Eigen::VectorXd& a,
            const Eigen::VectorXd& b) const override;

        std::unique_ptr<KernelFunction> copy() const override;

        std::vector<ParameterHandlerPtr> getParameters() const override;

    private:
        Parameter teta0;
        Parameter teta1;
    };
} // namespace gauss::gp
