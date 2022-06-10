file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

add_executable(KernelsViz ${SRC_FILES})

include(FetchContent)
FetchContent_Declare(
json_lib
GIT_REPOSITORY https://github.com/nlohmann/json.git
GIT_TAG        v3.10.5
)
FetchContent_MakeAvailable(json_lib)

target_link_libraries(KernelsViz
    PUBLIC GaussianProcess
    PRIVATE nlohmann_json::nlohmann_json
)
add_custom_command(TARGET KernelsViz PRE_BUILD
COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Visualizer.py ${CMAKE_CURRENT_BINARY_DIR}/Visualizer.py)
