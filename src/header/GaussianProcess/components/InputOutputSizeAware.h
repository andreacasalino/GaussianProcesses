/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianUtils/components/StateSpaceSizeAware.h>

namespace gauss::gp {
class InputOutputSizeAware : public StateSpaceSizeAware {
public:
  std::size_t getStateSpaceSize() const override {
    return getInputStateSpaceSize();
  }
  virtual std::size_t getInputStateSpaceSize() const = 0;
  virtual std::size_t getOutputStateSpaceSize() const = 0;

//protected:
//  InputOutputSizeAware(const std::size_t input_space_size,
//                       const std::size_t output_space_size)
//      : input_space_size(input_space_size),
//        output_space_size(output_space_size){};
//
//  std::size_t input_space_size;
//  std::size_t output_space_size;
};
} // namespace gauss::gp
