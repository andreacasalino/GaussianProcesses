
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

  const Eigen::MatrixXd &getYYtrain() const;

protected:
  YYMatrixTrain();

private:
  std::unique_ptr<SymmetricResizableMatrix> YYtrain;
};

class ResizableMatrix;

class YYMatrixPredict : virtual public TrainSetAware {
public:
  ~YYMatrixPredict();

  const Eigen::MatrixXd &getYYpredict() const;

protected:
  YYMatrixPredict();

private:
  std::unique_ptr<ResizableMatrix> YYpredict;
};
} // namespace gauss::gp
