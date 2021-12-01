include(FetchContent)
FetchContent_Declare(
  guass_utils
  GIT_REPOSITORY https://github.com/andreacasalino/Gaussian-Utils.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(guass_utils)

set(PROJECT_SHORTNAME GaussianProcess)

MakeLibrary(${PROJECT_SHORTNAME} header)

target_link_libraries(${PROJECT_SHORTNAME} PUBLIC
GaussianUtils
)

target_include_directories(${PROJECT_SHORTNAME} PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/src
)
