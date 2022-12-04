set(include_all ${include_all}
        ${PROJECT_SOURCE_DIR}/include
        )

file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(src_all ${src_all} ${_tmp})

file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}/*.cpp)
set(src_all ${src_all} ${_tmp})

string(TOLOWER ${PROJECT_NAME} LIB_NAME)

if (BUILD_SHARED_LIBS)
    add_library(${LIB_NAME} SHARED ${src_all})
else()
    add_library(${LIB_NAME} STATIC ${src_all})
endif ()


target_include_directories(${LIB_NAME} PRIVATE
        ${include_all}
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:${PROJECT_SOURCE_DIR}/include/${ENGINE_LOWER}>
        )

target_compile_definitions(${LIB_NAME} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_options(${LIB_NAME} PRIVATE
        -Wl,--warn-unresolved-symbols
        )

target_link_libraries(${LIB_NAME} PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        rknn_runtime
        pthread
        dl
        )
