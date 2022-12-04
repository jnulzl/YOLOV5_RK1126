//
// Created by lizhaoliang-os on 2020/6/23.
//

#ifndef ALG_DEFINE_H
#define ALG_DEFINE_H

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE
        #define ALG_BUILD_FOR_IOS
    #endif
#endif

#if defined(__ANDROID__)
    #include <android/log.h>
    #define ALG_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MNNJNI", format, ##__VA_ARGS__)
    #define ALG_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#else
    #define ALG_PRINT(format, ...) printf(format, ##__VA_ARGS__)
    #define ALG_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define ALG_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            ALG_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define ALG_ASSERT(x)                                        \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            ALG_ERROR("Error for %d\n", __LINE__);           \
        }                                                        \
    }
#endif

#define ALG_FUNC_PRINT(x) ALG_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define ALG_FUNC_PRINT_ALL(x, type) ALG_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define ALG_CHECK(success, log) \
if(!(success)){ \
ALG_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}


#if defined(_MSC_VER)
    #if defined(BUILDING_ALG_DLL)
        #define ALG_PUBLIC __declspec(dllexport)
    #elif defined(USING_ALG_DLL)
        #define ALG_PUBLIC __declspec(dllimport)
    #else
        #define ALG_PUBLIC
    #endif
#else
    #define ALG_PUBLIC __attribute__((visibility("default")))
#endif

#endif //ALG_DEFINE_H
