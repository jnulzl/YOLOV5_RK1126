set(demo_src detect_images_model_as_header_files.cpp)
get_filename_component(exe_name ${demo_src} NAME_WLE)

add_executable(${exe_name} ${demo_src})
target_include_directories(${exe_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

string(TOLOWER ${PROJECT_NAME} LIB_NAME)

target_compile_options(${exe_name} PRIVATE
        -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}"
        )
target_compile_definitions(${exe_name} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_options(${exe_name} PRIVATE
        -Wl,--no-undefined
        -Wl,--no-allow-shlib-undefined
        -Wl,--warn-unresolved-symbols
        )

if(BUILD_SHARED_LIBS)
    target_link_libraries(${exe_name} PRIVATE
            ${LIB_NAME}
            )
else()
    target_link_libraries(${exe_name} PRIVATE
            -Wl,--whole-archive ${LIB_NAME} -Wl,--no-whole-archive
            )
    target_link_options(${exe_name} PRIVATE
            -Wl,--allow-multiple-definition
            )
endif()

target_link_libraries(${exe_name} PRIVATE
        opencv_core${OpenCV_VERSION}
        opencv_highgui${OpenCV_VERSION}
        opencv_imgproc${OpenCV_VERSION}
        opencv_imgcodecs${OpenCV_VERSION}
        opencv_videoio${OpenCV_VERSION}
        opencv_video${OpenCV_VERSION}
        opencv_photo${OpenCV_VERSION}
        m
        )
