/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/CompositeKernelFunction.h>
#include <GaussianProcess/Error.h>

namespace gauss::gp {
  CompositeKernelFunction::CompositeKernelFunction(KernelFunctionPtr initial_element) {
      push_function(std::move(initial_element));
  }

  void CompositeKernelFunction::push_function(KernelFunctionPtr element) {
      if(nullptr == element) {
          throw gauss::gp::Error("empty element");
      }
      elements.emplace_back(std::move(element));
  }

  // evaluation should be reflexive: evaluate(a,b) = evaluate(b,a)
  double CompositeKernelFunction::evaluate(const Eigen::VectorXd &a,
                          const Eigen::VectorXd &b) const {
      double result = 0.0;
      for(const auto& element : elements) {
          result += element->evaluate(a,b);
      }
      return result;
  }

  std::unique_ptr<KernelFunction> CompositeKernelFunction::copy() const {
      auto it_element = elements.begin();
      std::unique_ptr<CompositeKernelFunction> result = std::make_unique<CompositeKernelFunction>((*it_element)->copy());
      ++it_element;
      for(it_element; it_element != elements.end(); ++it_element) {
          result->push_function((*it_element)->copy());
      }
      return result;
  };

  std::vector<ParameterHandlerPtr> CompositeKernelFunction::getParameters() const {
      std::vector<ParameterHandlerPtr> result;
      for(const auto& element : elements) {
          auto temp = element->getParameters();
          for(auto& param : temp) {
              result.emplace_back(std::move(param));
          }
      }
      return result;
  }
}
