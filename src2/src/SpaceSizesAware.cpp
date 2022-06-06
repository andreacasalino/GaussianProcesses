/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/SpaceSizesAware.h>

namespace gauss::gp {
SpaceSizesAware::SpaceSizesAware(const std::size_t in_size,
                                 const std::size_t out_size)
    : input_space_size(in_size), output_space_size(out_size) {
  if (0 == in_size) {
    throw Error{"Invalid input space size"};
  }
  if (0 == out_size) {
    throw Error{"Invalid output space size"};
  }
}
} // namespace gauss::gp
