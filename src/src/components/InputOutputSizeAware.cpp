/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/components/InputOutputSizeAware.h>
#include <GaussianProcess/Error.h>

namespace gauss::gp {
    InputOutputSizeAwareBase::InputOutputSizeAwareBase(
        const std::size_t input_space_size,
        const std::size_t output_space_size) {
        if ((0 == input_space_size) || (0 == output_space_size)) {
            throw Error("invalid space size");
        }
        this->input_space_size = input_space_size;
        this->output_space_size = output_space_size;
    }
} // namespace gauss::gp
