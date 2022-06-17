FECTH_JSON_LIB()

function(MAKE_VIZ NAME)
    add_executable(${NAME} ${NAME}.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/../samples/Ranges.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../samples/Ranges.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../samples/LogUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../samples/LogUtils.cpp
    )
    target_link_libraries(${NAME}
        PUBLIC GaussianProcess
        PRIVATE nlohmann_json::nlohmann_json
    )
    add_custom_command(TARGET ${NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.py ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.py)    
endfunction()

MAKE_VIZ(KernelsVisualizer)
MAKE_VIZ(LikelihoodVisualizer)
