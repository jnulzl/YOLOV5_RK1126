cmake_minimum_required(VERSION 3.15)
project(rv1126armv7)

set(CMAKE_SYSTEM_PROCESSOR arm) # arm aarch64
message(STATUS "CMAKE_SYSTEM_NAME      : " ${CMAKE_SYSTEM_NAME})
message(STATUS "CMAKE_SYSTEM_PROCESSOR : " ${CMAKE_SYSTEM_PROCESSOR})

set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon")
set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

# Select Device
set(DEVICE_LOWER cpu) # cpu cuda
string(TOUPPER ${DEVICE_LOWER} DEVICE_UPPER)
set(USE_${DEVICE_UPPER} ON)

# Select Engine
set(ENGINE_LOWER rv1126) # mnn
string(TOUPPER ${ENGINE_LOWER} ENGINE_UPPER)
set(USE_${ENGINE_UPPER} ON)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
##############Set output directory##############
set(OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME})
file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY})
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(LIBRARY_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH})

##############opencv##############
set(OpeCV_DIR ${PROJECT_SOURCE_DIR}/third_party/${CMAKE_SYSTEM_NAME}/opencv420)
set(OpeCV_INCLUDE ${OpeCV_DIR}/include ${OpeCV_DIR}/include/opencv4 ${OpeCV_DIR}/include/opencv4/opencv2)
set(OpeCV_LIB_PATH ${OpeCV_DIR}/lib)
set(OpenCV_VERSION "")

##############RK126##############
set(RKNN_DIR ${PROJECT_SOURCE_DIR}/third_party/${CMAKE_SYSTEM_NAME}/rv1126)
set(RKNN_LIB_PATH ${RKNN_DIR}/lib)

set(INCLUDE_PATH ${INCLUDE_PATH}
        ${RKNN_INCLUDE}
        ${OpeCV_INCLUDE}
        )
set(LIB_PATH ${LIB_PATH}
        ${RKNN_LIB_PATH}
        ${OpeCV_LIB_PATH}
        )

include_directories(${INCLUDE_PATH})
link_directories(${LIB_PATH})

add_subdirectory(src)
add_subdirectory(examples)
