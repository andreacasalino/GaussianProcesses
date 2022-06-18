message("Fetching dependencies")

include(FetchContent)
FetchContent_Declare(
  guass_utils
  GIT_REPOSITORY https://github.com/andreacasalino/Gaussian-Utils.git
  GIT_TAG        BetterEigenSupport
)
FetchContent_MakeAvailable(guass_utils)

FetchContent_Declare(
  trainers
  GIT_REPOSITORY https://github.com/andreacasalino/TrainingTools.git
  GIT_TAG        BetterEigenSupport
)
FetchContent_MakeAvailable(trainers)

set(PROJECT_SHORTNAME GaussianProcess)

MakeLibrary(${PROJECT_SHORTNAME} header)

target_link_libraries(${PROJECT_SHORTNAME} PUBLIC
GaussianUtils
TrainingTools
)

target_include_directories(${PROJECT_SHORTNAME} PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/src
)
