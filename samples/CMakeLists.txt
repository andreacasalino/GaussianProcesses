option(BUILD_SAMPLE_01 "Build Sample_01" ON)
if(BUILD_SAMPLE_01)
    add_executable(Sample01 01-Prediction-1d.cpp Utils.h)
    target_link_libraries(Sample01 PUBLIC GaussianProcess)	

    add_custom_command(TARGET Sample01 PRE_BUILD
	    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Visualize-1d.py ${CMAKE_CURRENT_BINARY_DIR}/Visualize-1d.py)
endif()

option(BUILD_SAMPLE_02 "Build Sample_02" ON)
if(BUILD_SAMPLE_02)
    add_executable(Sample02 02-Prediction-2d.cpp Utils.h)
    target_link_libraries(Sample02 PUBLIC GaussianProcess)	

    # add_custom_command(TARGET Sample01 PRE_BUILD
	    # COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Visualize-2d.py ${CMAKE_CURRENT_BINARY_DIR}/Visualize-2d.py)
endif()
