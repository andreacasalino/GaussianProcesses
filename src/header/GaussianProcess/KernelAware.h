/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>
#include <GaussianProcess/TrainSet.h>
#include <GaussianUtils/components/StateSpaceSizeAware.h>

namespace gauss::gp {
	class KernelAware
		: public StateSpaceSizeAware {
	public:
		virtual ~KernelAware() = default;

		KernelAware(const KernelAware& );
		KernelAware& operator=(const KernelAware&);

		KernelAware(KernelAware&&);
		KernelAware& operator=(KernelAware&&);

		void updateKernalFunction(KernelFunctionPtr new_kernel);

		void pushSample(const Eigen::VectorXd& input_sample, const Eigen::VectorXd& output_sample);
		template<typename IterableT>
		void pushSample(const IterableT& input_samples, const IterableT& output_samples);
		void clearSamples();

		std::size_t getStateSpaceSize() const override { return input_space_size; }
		std::size_t getOutputStateSpaceSize() const { return output_space_size; }

	protected:
		KernelAware(KernelFunctionPtr new_kernel, 
					const std::size_t input_space_size, 
					const std::size_t output_space_size);

		const TrainSet& getSamples() const { return *samples; };
		const Eigen::MatrixXd& getKernelInverse() const { return *kernel_inverse; };
		const Eigen::MatrixXd& getSamplesOutputMatrix() const { return *samples_output_matrix; };

	private:
		void recomputeKernel();
		void recomputeSamplesOutputMatrix();

		std::size_t input_space_size;
		std::size_t output_space_size;

		KernelFunctionPtr kernelFunction;
		std::unique_ptr<TrainSet> samples;
		std::unique_ptr<const Eigen::MatrixXd> kernel;
		std::unique_ptr<const Eigen::MatrixXd> kernel_inverse;
		std::unique_ptr<const Eigen::MatrixXd> samples_output_matrix;
	};
}
