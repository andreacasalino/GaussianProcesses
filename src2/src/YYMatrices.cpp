
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
        const auto samples = this->getTrainSet()->GetSamplesOutput();
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

class YYResizableMatrix : public ResizableMatrix {
public:
  using Emplacer = std::function<Eigen::VectorXd(const Eigen::Index)>;
  YYResizableMatrix(const Emplacer &emplacer) : emplacer(emplacer) {}

protected:
  Eigen::MatrixXd makeResized() const final {
    const auto size = getSize();
    const auto computed_size = getComputedSize();
    throw std::runtime_error{"TODO"};
    // Eigen::MatrixXd new_matrix = Eigen::MatrixXd::Zero(size, size);
    // if (0 == computed_size) {
    //   compute_symmetric_block(new_matrix, emplacer, IndexInterval{0, size});
    // } else {
    //   new_matrix.block(0, 0, computed_size, computed_size) =
    //       getComputedPortion();
    //   compute_symmetric_block(new_matrix, emplacer,
    //                           IndexInterval{computed_size, size});
    //   compute_asymmetric_block(new_matrix, emplacer,
    //                            IndexInterval{0, computed_size},
    //                            IndexInterval{computed_size, size});
    //   new_matrix.block(computed_size, 0, size - computed_size, computed_size)
    //   =
    //       new_matrix
    //           .block(0, computed_size, computed_size, size - computed_size)
    //           .transpose();
    // }
    // return new_matrix;
  }

private:
  const Emplacer emplacer;
};

YYMatrixPredict::YYMatrixPredict() {
  YYpredict =
      std::make_unique<YYResizableMatrix>([this](const Eigen::Index index) {
        const auto samples = this->getTrainSet()->GetSamplesOutput();
        return samples[static_cast<std::size_t>(index)];
      });
}

const Eigen::MatrixXd &YYMatrixPredict::getYYpredict() const {
  YYpredict->resize(getTrainSet().GetSamplesInput().size());
  return YYpredict->access();
}
} // namespace gauss::gp
