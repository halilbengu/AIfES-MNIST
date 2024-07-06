cmake_minimum_required(VERSION 3.5.0)
project(AIfES_CNN VERSION 0.1.0 LANGUAGES C)

set( executableOutputPath ${PROJECT_SOURCE_DIR}/build/ )

include_directories(${CMAKE_SOURCE_DIR})
file(GLOB_RECURSE AIfES_SRC_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/basic/*.c"
    "${PROJECT_SOURCE_DIR}/cnn/*.c"
    "${PROJECT_SOURCE_DIR}/core/*.c"
)

add_executable(AIfES_CNN main.c ${AIfES_SRC_FILES})