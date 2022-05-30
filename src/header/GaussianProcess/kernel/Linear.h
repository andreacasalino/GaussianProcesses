// /**
//  * Author:    Andrea Casalino
//  * Created:   29.11.2021
//  *
//  * report any bug to andrecasa91@gmail.com.
//  **/

// #pragma once

// #include <GaussianProcess/kernel/KernelFunction.h>
// #include <vector>

// namespace gauss::gp {
// /**
//  * @brief Kernel function k(x1, x2) assumed equal to:
//  * teta0^2 + teta1^2 * (x1 - mean).dot( (x2 -mean) )
//  */
// class LinearFunction : public KernelFunction {
// public:
//   LinearFunction(const double teta0, const double teta1,
//                  const std::size_t space_size);
//   LinearFunction(const double teta0, const double teta1,
//                  const Eigen::VectorXd &mean);

//   double evaluate(const Eigen::VectorXd &a,
//                   const Eigen::VectorXd &b) const override;

//   std::unique_ptr<KernelFunction> copy() const override;

//   std::vector<ParameterHandlerPtr> getParameters() const override;

// protected:
//   LinearFunction(const LinearFunction &o);

// private:
//   Parameter teta0;
//   Parameter teta1;
//   std::vector<Parameter> mean;
// };
// } // namespace gauss::gp
