/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianUtils/TrainSet.h>

namespace gauss::gp {
    class GaussianProcessTrainSet {
    public:

        const TrainSet& GetSamplesInput() const { return input; };
        const TrainSet& GetSamplesOutput() const { return output; };

    private:
        gauss::TrainSet input;
        gauss::TrainSet output;
    };
}
