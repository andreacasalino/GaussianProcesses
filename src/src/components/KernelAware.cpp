/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/KernelAware.h>
#include <GaussianProcess/Error.h>
#include <GaussianUtils/Utils.h>

namespace gauss::gp {
    KernelAware::KernelAware(KernelFunctionPtr new_kernel,
        const std::size_t input_space_size,
        const std::size_t output_space_size) {
        if (nullptr == new_kernel) {
            throw Error("empty kernel function");
        }
        kernelFunction = std::move(new_kernel);
        this->input_space_size = input_space_size;
        this->output_space_size = output_space_size;
    }

    namespace {
        struct Indices {
            const std::size_t start;
            const std::size_t end;
        };
        Eigen::MatrixXd compute_kernel(const std::vector<Eigen::VectorXd>& samples, const KernelFunction& function,
                                       const Indices& rows, const Indices& cols) {
            Eigen::MatrixXd  result(rows.end - rows.start, cols.end - cols.start);
            std::size_t c;
            for (std::size_t r = rows.start; r < rows.end; ++r) {
                for (c = cols.start; c < cols.end; ++c) {
                    result(r, c) = function.evaluate(samples[r], samples[c]);
                }
            }
            return result;
        }
    }

    void KernelAware::recomputeKernel() {
        const auto& input_samples = samples->GetSamplesInput().GetSamples();
        if (nullptr == kernel) {
            kernel = std::make_unique<Eigen::MatrixXd>(compute_kernel(input_samples, *kernelFunction, 
                                                                      Indices{0, input_samples.size()}, Indices{ 0, input_samples.size() }));
        }
        else {
            Indices old_indices = Indices{0, kernel->rows()};
            Indices new_indices = Indices{kernel->rows(), input_samples.size()};
            Eigen::MatrixXd K_old_new = compute_kernel(input_samples, *kernelFunction, old_indices, new_indices);
            Eigen::MatrixXd K_new_new = compute_kernel(input_samples, *kernelFunction, new_indices, new_indices);
            auto new_kernel = std::make_unique<Eigen::MatrixXd>(input_samples.size(), input_samples.size());
            *new_kernel << *kernel, K_old_new,
                            K_old_new.transpose(), K_new_new;
            throw 0; // check again computation above
            kernel = std::move(new_kernel);
        }
        kernel_inverse.reset(new Eigen::MatrixXd(computeCovarianceInvert(*kernel)));
    }

    namespace {
        Eigen::MatrixXd compute_output_matrix(const std::vector<Eigen::VectorXd>& samples, const Indices& indices) {
            Eigen::MatrixXd  result(indices.end - indices.start, samples.front().size());
            for (std::size_t i = indices.start; i < indices.end; ++i) {
                result.row(i) = samples[i];
            }
            return result;
        }
    }

    void KernelAware::recomputeSamplesOutputMatrix() {
        const auto& output_samples = samples->GetSamplesOutput().GetSamples();
        if (nullptr == samples_output_matrix) {
            samples_output_matrix = std::make_unique<Eigen::MatrixXd>(compute_output_matrix(output_samples, Indices{0, output_samples .size()}));
        }
        else {
            auto new_output_matrix = std::make_unique<Eigen::MatrixXd>(output_samples.size(), output_space_size);
            *new_output_matrix << *samples_output_matrix, compute_output_matrix(output_samples, Indices{ samples_output_matrix->rows(), output_samples.size() });
            throw 0; // check again computation above
            samples_output_matrix = std::move(new_output_matrix);
        }
    }

    void KernelAware::updateKernalFunction(KernelFunctionPtr new_kernel) {
        if (nullptr == new_kernel) {
            throw Error("empty kernel function");
        }
        kernelFunction = std::move(new_kernel);
        kernel.reset();
        recomputeKernel();
    }
}
