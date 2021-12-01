/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/GaussianProcessVectorial.h>

namespace gauss::gp {
    GaussianProcessVectorial::GaussianProcessVectorial(KernelFunctionPtr kernel, const std::size_t input_space_size, const std::size_t output_space_size)
        : GaussianProcessBase(std::move(kernel), input_space_size, output_space_size) {
    }

    std::vector<gauss::GaussianDistribution> GaussianProcessVectorial::predict(const Eigen::VectorXd& point) const {
        Eigen::VectorXd prediction_mean;
        double prediction_covariance;
        GaussianProcessBase::predict(point, prediction_mean, prediction_covariance);
        Eigen::MatrixXd prediction_covariance_mat(1, 1);
        prediction_covariance_mat << prediction_covariance;
        std::vector<gauss::GaussianDistribution> result;
        result.reserve(prediction_mean.size());
        for (Eigen::Index k = 0; k < prediction_mean.size(); ++k) {
            Eigen::VectorXd temp(1);
            temp << prediction_mean(k);
            result.emplace_back(temp, prediction_covariance_mat);
        }
        return result;
    }
}
