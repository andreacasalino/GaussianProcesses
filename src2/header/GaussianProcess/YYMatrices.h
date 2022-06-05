
/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/TrainSet.h>

namespace gauss::gp {
class SymmetricResizableMatrix;

class YYMatrixTrain : virtual public TrainSetAware {
public:
  ~YYMatrixTrain();

protected:
  YYMatrixTrain();

  const Eigen::MatrixXd &getYYtrain() const;

private:
  std::unique_ptr<SymmetricResizableMatrix> YYtrain;
};

class YYMatrixPredict : virtual public TrainSetAware {
public:
  ~YYMatrixPredict();

protected:
  YYMatrixPredict();

  const Eigen::MatrixXd &getYYpredict() const;

private:
  class YYResizableMatrix;
  std::unique_ptr<YYResizableMatrix> YYpredict;
};
} // namespace gauss::gp
