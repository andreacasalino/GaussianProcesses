macro(MAKE_SAMPLE NAME SCRIPT)
    add_executable(${NAME} ${NAME}.cpp Utils.h)
    target_link_libraries(${NAME} PUBLIC GaussianProcess)	

    add_custom_command(TARGET ${NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT}.py ${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT}.py)    
endmacro()

MAKE_SAMPLE(Sample01-R1-function Visualize01)
MAKE_SAMPLE(Sample02-R2-function Visualize02)
