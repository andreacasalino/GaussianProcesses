
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
  ~YYMatrixTrain() override;

  Eigen::MatrixXd getYYtrain() const { return getYYtrain_(); };

protected:
  YYMatrixTrain();

  const Eigen::MatrixXd &getYYtrain_() const;

private:
  std::unique_ptr<SymmetricResizableMatrix> YYtrain;
};

class ResizableMatrix;

class YYMatrixPredict : virtual public TrainSetAware {
public:
  ~YYMatrixPredict() override;

  Eigen::MatrixXd getYYpredict() const { return getYYpredict_(); };

protected:
  YYMatrixPredict();

  const Eigen::MatrixXd &getYYpredict_() const;

private:
  std::unique_ptr<ResizableMatrix> YYpredict;
};
} // namespace gauss::gp
