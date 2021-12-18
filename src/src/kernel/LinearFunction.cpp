/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/LinearFunction.h>

namespace gauss::gp {
    namespace {
        class Teta0Handler : public ParameterHandler {
        public:
            Teta0Handler(const Parameter& teta0) : ParameterHandler(teta0) {};

            double evaluate_gradient(const Eigen::VectorXd &a,
                                    const Eigen::VectorXd &b) const override {
                return 1.0;
            };
        };

        class Teta1Handler : public ParameterHandler {
        public:
            Teta1Handler(const Parameter& teta1) : ParameterHandler(teta1) {};

            double evaluate_gradient(const Eigen::VectorXd &a,
                                    const Eigen::VectorXd &b) const override {
                return a.dot(b);
            };
        };
    }

    LinearFunction::LinearFunction(const double teta0, const double teta1) {
        this->teta0 = std::make_shared<double>(teta0);
        this->teta1 = std::make_shared<double>(teta1);
    }

    double LinearFunction::evaluate(const Eigen::VectorXd& a,
        const Eigen::VectorXd& b) const {
        return *teta0 + *teta1 * a.dot(b);
    };

    std::unique_ptr<KernelFunction> LinearFunction::copy() const {
        return std::make_unique<LinearFunction>(*teta0, *teta1);
    };

    std::vector<ParameterHandlerPtr> LinearFunction::getParameters() const {
        std::vector<ParameterHandlerPtr> result;
        result.reserve(2);
        result.emplace_back(std::make_unique<Teta0Handler>(teta0));
        result.emplace_back(std::make_unique<Teta0Handler>(teta1));
        return result;
    };
}
