FECTH_JSON_LIB()

macro(ADD_SCRIPT NAME SCRIPT)
    add_custom_command(TARGET ${NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT}.py ${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT}.py)
endmacro()


macro(MAKE_SAMPLE NAME SCRIPT)
    add_executable(${NAME} ${NAME}.cpp Utils.h Utils.cpp)
    target_link_libraries(${NAME} PUBLIC 
        GaussianProcess
        nlohmann_json::nlohmann_json
    )	

    ADD_SCRIPT(${NAME} ${SCRIPT})
endmacro()

MAKE_SAMPLE(Sample01-1D-functions Visualize-1D)
MAKE_SAMPLE(Sample02-2D-functions Visualize-2D)
