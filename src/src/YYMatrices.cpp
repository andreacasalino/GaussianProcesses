
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

const Eigen::MatrixXd &YYMatrixTrain::getYYtrain_() const {
  YYtrain->resize(getTrainSet().GetSamplesInput().size());
  return YYtrain->access();
}

namespace {
class YYMatrixPredict_ : public ResizableMatrix {
public:
  YYMatrixPredict_(const TrainSetAware &source) : source(source) {}

protected:
  Eigen::MatrixXd makeResized() const final {
    const auto size = getSize();
    const auto computed_size = getComputedSize();
    const auto &train_set = source.getTrainSet();
    const auto &samples_out = train_set.GetSamplesOutput();
    Eigen::MatrixXd result = Eigen::MatrixXd{
        train_set.getOutputStateSpaceSize(), samples_out.size()};
    for (Eigen::Index c = 0; c < size; ++c) {
      result.col(c) = samples_out[c];
    }
    return result;
  }

private:
  const TrainSetAware &source;
};
} // namespace

YYMatrixPredict::YYMatrixPredict() {
  YYpredict = std::make_unique<YYMatrixPredict_>(*this);
}

YYMatrixPredict::~YYMatrixPredict() = default;

const Eigen::MatrixXd &YYMatrixPredict::getYYpredict_() const {
  YYpredict->resize(getTrainSet().GetSamplesInput().size());
  return YYpredict->access();
}
} // namespace gauss::gp
