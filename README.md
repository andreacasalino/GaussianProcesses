![binaries_compilation](https://github.com/andreacasalino/GaussianProcesses/actions/workflows/installArtifacts.yml/badge.svg)
![binaries_compilation](https://github.com/andreacasalino/GaussianProcesses/actions/workflows/runTests.yml/badge.svg)

This libary contains the functionalities required to train and handle **Gaussian Processes**, aka **GP**.
If you believe to be not really familiar with this object, have a look at these useful tutorials:
-[Gaussian Processes](https://www.youtube.com/watch?v=UBDgSHPxVME&t=794s)
-[Easy introduction to gaussian process regression](https://www.youtube.com/watch?v=iDzaoEwd0N0)

The **GP** hyperparameters can be trained with the gradient descend approaches that are part of this package.

This package is completely **cross-platform**: use [CMake](https://cmake.org) to configure the project containig the libary and some samples.

This library uses [**Eigen**](https://gitlab.com/libeigen/eigen) as internal linear algebra engine. 
**Eigen** is by default [fetched](https://cmake.org/cmake/help/latest/module/FetchContent.html) and copied by **CMake** from the latest version on the official **Eigen** repository.
However, you can also use a local version, by [setting](https://www.youtube.com/watch?v=LxHV-KNEG3k&t=1s) the **CMake** option **EIGEN_INSTALL_FOLDER** equal to the root folder storing the local **Eigen** you want to use.

If you have found this library useful, take the time to leave a star ;)