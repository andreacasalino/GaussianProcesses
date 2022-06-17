/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>

namespace gauss::gp {
const double SuspiciousCovarianceError::COVARIANCE_TOLLERANCE = -1e-4;

SuspiciousCovarianceError::SuspiciousCovarianceError(const std::string &what)
    : Error(what + ": check the correctness of the function or its "
                   "weights function are badly set") {}
} // namespace gauss::gp
