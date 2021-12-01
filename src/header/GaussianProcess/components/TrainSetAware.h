/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/TrainSet.h>
#include <memory>

namespace gauss::gp {
class TrainSetAware {
public:
  virtual ~TrainSetAware() = default;

protected:
  TrainSetAware() = default;

  std::unique_ptr<TrainSet> samples;
};
} // namespace gauss::gp
