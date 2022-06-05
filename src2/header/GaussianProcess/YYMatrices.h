
/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/TrainSet.h>
#include <memory>

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

class YYResizableMatrixPredict;
class YYMatrixPredict : virtual public TrainSetAware {
public:
  ~YYMatrixPredict();

protected:
  YYMatrixPredict();

  const Eigen::MatrixXd &getYYpredict() const;

private:
  std::unique_ptr<YYResizableMatrixPredict> YYpredict;
};
} // namespace gauss::gp
