/**
 * Author:    Andrea Casalino
 * Created:   24.12.2019
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <stdexcept>

namespace gauss::gp {
/** @brief A runtime error that can be raised by objects in this namespace
 */
class Error : public std::runtime_error {
public:
  explicit Error(const std::string &what) : std::runtime_error(what){};
};

/** @brief This error is raised when a bad covariance matrix for the gaussian
 * process is obtained. There might be many reasons why this happens: a bugged
 * kernel function custom implementation, too low or big hyperparameters for the
 * kernel function, etc...
 */
class SuspiciousCovarianceError : public Error {
public:
  SuspiciousCovarianceError(const std::string &what);

  static const double COVARIANCE_TOLLERANCE;
};
} // namespace gauss::gp
