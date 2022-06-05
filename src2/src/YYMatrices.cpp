
/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>
#include <GaussianProcess/Error.h>
#include <GaussianProcess/YYMatrices.h>

namespace gauss::gp {
YYMatrixTrain::YYMatrixTrain() {
  YYtrain = std::make_unique<SymmetricResizableMatrix>(
      [this](const Eigen::Index row, const Eigen::Index col) {
        const auto samples = this->getTrainSet().GetSamplesOutput();
        const auto &y1 = samples[static_cast<std::size_t>(row)];
        const auto &y2 = samples[static_cast<std::size_t>(col)];
        return y1.dot(y2);
      });
}

YYMatrixTrain::~YYMatrixTrain() = default;

const Eigen::MatrixXd &YYMatrixTrain::getYYtrain() const {
  YYtrain->resize(getTrainSet().GetSamplesInput().size());
  return YYtrain->access();
}

class YYResizableMatrixPredict : public ResizableMatrix {
public:
  YYResizableMatrixPredict(const TrainSet &samples) : samples(samples) {}

protected:
  Eigen::MatrixXd makeResized() const final {
    const auto size = getSize();
    const auto computed_size = getComputedSize();
    const auto &samples_in = samples.GetSamplesInput();
    Eigen::MatrixXd result =
        Eigen::MatrixXd{samples.getInputStateSpaceSize(), samples_in.size()};
    result.block(0, 0, samples.getInputStateSpaceSize(), computed_size) =
        getComputedPortion();
    for (Eigen::Index c = computed_size; c < size; ++c) {
      result.col(c) = samples_in[c];
    }
    return result;
  }

private:
  const TrainSet &samples;
};

YYMatrixPredict::YYMatrixPredict() {
  YYpredict = std::make_unique<YYResizableMatrixPredict>(getTrainSet());
}

YYMatrixPredict::~YYMatrixPredict() = default;

const Eigen::MatrixXd &YYMatrixPredict::getYYpredict() const {
  YYpredict->resize(getTrainSet().GetSamplesInput().size());
  return YYpredict->access();
}
} // namespace gauss::gp
