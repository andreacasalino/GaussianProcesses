/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/TrainSet.h>
#include <GaussianProcess/Error.h>

namespace gauss::gp {
    namespace {
        Eigen::VectorXd get_slice(const Eigen::VectorXd& source, const Eigen::Index& start, const Eigen::Index& end) {
            Eigen::VectorXd result(end-start);
            Eigen::Index i = 0;
            for (auto p = start; p < end; ++p, ++i) {
                result(i) = source(p);
            }
            return result;
        }
    }

    TrainSet::TrainSet(const std::string& file_to_read, const std::size_t input_space_size) {
        gauss::TrainSet temp(file_to_read);
        if (static_cast<std::size_t>(temp.GetSamples().front().size()) < input_space_size) {
            throw gauss::gp::Error("Invalid size of samples in parsed files");
        }
        auto it = temp.GetSamples().begin();
        input = std::make_unique<gauss::TrainSet>(get_slice (*it, 0, input_space_size));
        output = std::make_unique<gauss::TrainSet>(get_slice(*it, input_space_size + 1, it->size()));
        ++it;
        for (it; it != temp.GetSamples().end(); ++it) {
            *input += get_slice(*it, 0, input_space_size);
            *output += get_slice(*it, input_space_size + 1, it->size());
        }
    }

    TrainSet::TrainSet(gauss::TrainSet input_samples, gauss::TrainSet output_samples) {
        this->input = std::make_unique<gauss::TrainSet>(std::move(input_samples));
        this->output = std::make_unique<gauss::TrainSet>(std::move(output_samples));
    }

    void TrainSet::operator+=(const Eigen::VectorXd& sample) {
        if (sample.size() != (getInputStateSpaceSize() + getOutputStateSpaceSize())) {
            throw gauss::gp::Error("Invalid new sample");
        }
        *input += get_slice(sample, 0, getInputStateSpaceSize());
        *output += get_slice(sample, getInputStateSpaceSize() + 1, sample.size());

    }
}
