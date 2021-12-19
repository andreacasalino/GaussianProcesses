option(BUILD_SAMPLE_01 "Build Sample_01" ON)
if(BUILD_SAMPLE_01)
    add_executable(Sample01 01-Prediction-1d.cpp Utils.h)
    target_link_libraries(Sample01 PUBLIC GaussianProcess)	

    # add_custom_command(TARGET Sample01 PRE_BUILD
	    # COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Visualize01.py ${CMAKE_CURRENT_BINARY_DIR}/Visualize01.py)
endif()
