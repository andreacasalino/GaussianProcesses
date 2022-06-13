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

MAKE_SAMPLE(Sample01-1D-functions Visualize)
# ADD_SCRIPT(Sample01-R1-function VisualizeEvolution)

# MAKE_SAMPLE(Sample02-R2-functionA Visualize02)
# MAKE_SAMPLE(Sample03-R2-functionB Visualize02)
