message("Fetching dependencies")

include(FetchContent)
FetchContent_Declare(
  guass_utils
  GIT_REPOSITORY https://github.com/andreacasalino/Gaussian-Utils.git
  GIT_TAG        58b3f207cf6df73a0b7cdc8bde933b06d20ab502
)
FetchContent_MakeAvailable(guass_utils)

FetchContent_Declare(
  trainers
  GIT_REPOSITORY https://github.com/andreacasalino/TrainingTools.git
  GIT_TAG        5acb40e087f3150c206b756d72d51e787e752414
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
