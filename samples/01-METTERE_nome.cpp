#include <GaussianProcess/GaussianProcess.h>
#include <GaussianUtils/GaussianDistributionFactory.h>
#include <GaussianProcess/kernel/RadialBasisFunction.h>
#include <iostream>

int main() {
	const std::size_t samples_numb = 5;
	auto input_generator = gauss::GaussianDistributionFactory{ 5 }.makeRandomModel();
	auto output_generator = gauss::GaussianDistributionFactory{ 1 }.makeRandomModel();
	gauss::gp::TrainSet samples(input_generator->drawSamples(samples_numb), output_generator->drawSamples(samples_numb));

	gauss::gp::GaussianProcess process(std::make_unique<gauss::gp::RadialBasisFunction>(),
										std::move(samples));

	std::cout << process.getCovariance() << std::endl << std::endl << std::endl;
	std::cout << process.getCovarianceInv() << std::endl << std::endl << std::endl;

	auto point = process.getTrainSet()->GetSamplesInput().GetSamples().front();
	{
		Eigen::VectorXd delta(point.size());
		delta.setRandom();
		point += delta;
	}
	auto prediction = process.predict(point);


	return EXIT_SUCCESS;
}