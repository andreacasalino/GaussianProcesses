/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianUtils/TrainSet.h>

namespace gauss::gp {
    class TrainSet {
    public:

        void push(const Eigen::VectorXd& input_sample, const Eigen::VectorXd& output_sample);

        const gauss::TrainSet& GetSamplesInput() const { return input; };
        const gauss::TrainSet& GetSamplesOutput() const { return output; };

    private:
        gauss::TrainSet input;
        gauss::TrainSet output;
    };
}
