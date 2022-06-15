#include <GaussianProcess/kernel/Linear.h>
#include <GaussianProcess/kernel/PeriodicFunction.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace gauss::gp {
nlohmann::json make_kernel_viz_log(const std::size_t samples_numb,
                                   const KernelFunction &kernel,
                                   const std::string &title,
                                   const std::string &tag);
} // namespace gauss::gp

int main() {
  const std::size_t size = 100;

  struct KernelAndTitle {
    gauss::gp::KernelFunctionPtr kernel;
    const std::string title;
  };

  std::unordered_map<std::string, std::vector<KernelAndTitle>> cases;

  ///////////// squared exponential function /////////////
  {
    auto &tag = cases["exponential"];
    tag.push_back(KernelAndTitle{
        std::make_unique<gauss::gp::SquaredExponential>(1.0, 0.1),
        "squared exponential d = 0.1"});
    tag.push_back(KernelAndTitle{
        std::make_unique<gauss::gp::SquaredExponential>(1.0, 0.5),
        "squared exponential d = 0.5"});
    tag.push_back(KernelAndTitle{
        std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0),
        "squared exponential d = 1.0"});
  }

  ///////////// periodic function /////////////
  {
    auto &tag = cases["periodic"];
    tag.push_back(KernelAndTitle{
        std::make_unique<gauss::gp::PeriodicFunction>(1.0, 0.5, 0.1),
        "periodic d = 0.5 p = 0.1"});
    tag.push_back(KernelAndTitle{
        std::make_unique<gauss::gp::PeriodicFunction>(1.0, 0.5, 0.5),
        "periodic d = 0.5 p = 0.5"});
    tag.push_back(KernelAndTitle{
        std::make_unique<gauss::gp::PeriodicFunction>(1.0, 0.5, 1.0),
        "periodic d = 0.5 p = 1.0"});
    tag.push_back(KernelAndTitle{
        std::make_unique<gauss::gp::PeriodicFunction>(1.0, 0.1, 0.5),
        "periodic d = 0.1 p = 0.5"});
    tag.push_back(KernelAndTitle{
        std::make_unique<gauss::gp::PeriodicFunction>(1.0, 1.0, 0.5),
        "periodic d = 1.0 p = 0.5"});
  }

  ///////////// linear function /////////////
  {
    auto &tag = cases["linear"];
    tag.push_back(
        KernelAndTitle{std::make_unique<gauss::gp::LinearFunction>(1.0, 1.0, 1),
                       "linear function mean = 0"});
    tag.push_back(KernelAndTitle{std::make_unique<gauss::gp::LinearFunction>(
                                     1.0, 1.0, -Eigen::VectorXd::Ones(1) * 0.7),
                                 "linear function mean = -0.7"});
    tag.push_back(KernelAndTitle{std::make_unique<gauss::gp::LinearFunction>(
                                     1.0, 1.0, Eigen::VectorXd::Ones(1) * 0.7),
                                 "linear function mean = 0.7"});
  }

  nlohmann::json kernels_log = nlohmann::json::array();
  for (const auto &[tag, data] : cases) {
    std::cout << "<========== " << tag << " ==========>" << std::endl;
    for (const auto &[function, title] : data) {
      std::cout << title;
      kernels_log.push_back(
          gauss::gp::make_kernel_viz_log(size, *function, title, tag));
      std::cout << " done" << std::endl;
    }
  }
  std::ofstream stream("kernels_log.json");
  stream << kernels_log.dump();

  return EXIT_SUCCESS;
}

namespace gauss::gp {
std::vector<double>
convert_samples(const std::vector<Eigen::VectorXd> &samples) {
  std::vector<double> result;
  result.reserve(samples.size());
  for (const auto &sample : samples) {
    result.push_back(sample(0));
  }
  return result;
}

namespace {
std::vector<Eigen::VectorXd> make_equi_samples(const std::size_t samples_numb) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples_numb);
  double val = -1.0;
  const double delta = 2.0 / static_cast<double>(samples_numb - 1);
  for (std::size_t k = 0; k < samples_numb; ++k) {
    auto &new_sample = result.emplace_back(1);
    new_sample << val;
    val += delta;
  }
  return result;
}
} // namespace

nlohmann::json make_kernel_viz_log(const std::size_t samples_numb,
                                   const KernelFunction &kernel,
                                   const std::string &title,
                                   const std::string &tag) {
  const auto samples = make_equi_samples(samples_numb);

  std::vector<std::vector<double>> kernel_matrix;
  kernel_matrix.reserve(samples_numb);
  for (Eigen::Index r = 0; r < samples_numb; ++r) {
    auto &row = kernel_matrix.emplace_back();
    row.reserve(samples_numb);
    for (Eigen::Index c = 0; c < samples_numb; ++c) {
      row.push_back(kernel.evaluate(samples[r], samples[c]));
    }
  }

  nlohmann::json recipient;
  recipient["samples"] = convert_samples(samples);
  recipient["kernel"] = kernel_matrix;
  recipient["title"] = title;
  recipient["tag"] = tag;
  return recipient;
}
} // namespace gauss::gp