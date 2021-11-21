include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(eigen)
find_package (Eigen3 REQUIRED NO_MODULE)

set(PROJECT_SHORTNAME "GaussianProcess")

MakeLibrary(${PROJECT_SHORTNAME} header)
target_link_libraries (${PROJECT_SHORTNAME}
PUBLIC 
	Eigen3::Eigen
)
