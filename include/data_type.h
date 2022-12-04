//
// Created by jnulzl on 2020/5/24.
//

#ifndef YOLOV5_DATA_TYPE_H
#define YOLOV5_DATA_TYPE_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>


#define ANY_POINTER_CAST(impl, T) reinterpret_cast<T*>(impl)
typedef void* AW_ANY_POINTER;

struct BaseConfig {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string weights_path;
    std::string deploy_path;
    uint8_t* model_buffer;
    size_t model_size;
    float means[3];
    float scales[3];
    int mean_length;
    int net_inp_channels;
    int net_inp_width;
    int net_inp_height;
    int num_threads = 4;
#ifdef USE_CUDA
    int batch_size = 1;
    int device_id = 0;
    #ifdef USE_TENSORRT
        int dlaCore = -1;
        bool fp16 = false;
        bool int8 = false;
    #endif
#elif USE_MTK
    bool allow_fp16 = false;
    std::string cache_dir = ""; //Recommand to use /data/data/your_apk_name/cache/
#endif
};

struct YoloConfig : public BaseConfig {
    float conf_thres;
    float nms_thresh;
    std::vector<int> strides;
    std::vector<std::vector<float>> anchor_grids;
};

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef enum:int
{
    IMG_BGR = 0,
    IMG_RGB = 1,
    IMG_GRAY = 2,
    IMG_BGRA32 = 3,
}InputDataType;

template <typename DATATYPE>
struct _Rect
{
    DATATYPE xmin;
    DATATYPE ymin;
    DATATYPE xmax;
    DATATYPE ymax;

    DATATYPE width;
    DATATYPE height;
};
typedef _Rect<float> RectFloat;
typedef _Rect<int> RectInt;

template <typename DATATYPE>
struct _Point
{
    DATATYPE x;
    DATATYPE y;
};
typedef _Point<float> PointFloat;
typedef _Point<int> PointInt;



#define PI (3.141592653589793)


#endif //YOLOV5_DATA_TYPE_H
