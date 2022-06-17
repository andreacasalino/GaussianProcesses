/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <cstdlib>

namespace gauss::gp {
class SpaceSizesAware {
public:
  std::size_t getInputStateSpaceSize() const { return input_space_size; }
  std::size_t getOutputStateSpaceSize() const { return output_space_size; }

protected:
  SpaceSizesAware(const std::size_t in_size, const std::size_t out_size);

private:
  const std::size_t input_space_size;
  const std::size_t output_space_size;
};
} // namespace gauss::gp
