#include "opencv2/opencv.hpp"
#include "Module_yolov5_impl.h"
#include "post_process.h"
#include "debug.h"

CModule_yolov5_impl::CModule_yolov5_impl() {}
CModule_yolov5_impl::~CModule_yolov5_impl() 
{
    reinterpret_cast<cv::Mat*>(des_mat_)->release();
    delete reinterpret_cast<cv::Mat*>(des_mat_);
#ifdef ALG_DEBUG
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_yolov5_impl::init(const YoloConfig& config)
{
	config_ = config;
    des_mat_ = new cv::Mat(config_.net_inp_height, config_.net_inp_width, CV_8UC3);
    engine_init();
}

void CModule_yolov5_impl::preprocess()
{
//    std::cout << "This is default preprocess!!!!! " << std::endl;
}

void CModule_yolov5_impl::process(const void* mat)
{
#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
#endif
    img_height_ = reinterpret_cast<const cv::Mat*>(mat)->rows;
    img_width_ = reinterpret_cast<const cv::Mat*>(mat)->cols;

    yolo_resize_with_opencv(reinterpret_cast<const cv::Mat*>(mat), config_.net_inp_height, config_.net_inp_width);
    src_resize_ptr_ = reinterpret_cast<cv::Mat*>(des_mat_)->data; //bgr

#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_nms = std::chrono::system_clock::now();
    std::printf("preprocess time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count());
    begin_time_nms = std::chrono::system_clock::now();
#endif

    engine_run();
#ifdef ALG_DEBUG
    end_time_nms = std::chrono::system_clock::now();
    std::printf("engine_run time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count());
    begin_time_nms = std::chrono::system_clock::now();
#endif

    int num_obj = data_out_.size() / step_each_obj_;
    non_max_suppression(data_out_.data(), num_obj, step_each_obj_, config_.conf_thres, config_.nms_thresh, boxs_);
    postprocess(boxs_, config_.net_inp_height, config_.net_inp_width, img_height_, img_width_);
#ifdef ALG_DEBUG
    end_time_nms = std::chrono::system_clock::now();
    std::printf("postprocess2 time1 %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count());
#endif
}

const std::vector<BoxInfo>& CModule_yolov5_impl::get_result()
{
    engine_post(config_.output_names);
    return boxs_;
}

void CModule_yolov5_impl::yolo_resize_with_opencv(const void* mat_ptr, int des_height, int des_width)
{
    assert(des_height == des_width);

    cv::Mat des = *reinterpret_cast<cv::Mat*>(des_mat_);
    cv::Mat src = *reinterpret_cast<const cv::Mat*>(mat_ptr);
    int src_height = src.rows;
    int src_width  = src.cols;
    if (src_height == src_width)
    {
        cv::resize(src, des , cv::Size(des_width, des_height), 0, 0, cv::INTER_NEAREST);
        return;
    }

    float scale_wh = des_height / std::fmax(1.0f * src_height, 1.0f * src_width);
    int src_new_height = scale_wh * src_height;
    int src_new_width = scale_wh * src_width;

    cv::Mat src_resize;
    cv::resize(src, src_resize, cv::Size(src_new_width, src_new_height), 0, 0, cv::INTER_NEAREST);;

    cv::Rect2i roi;
    roi.height = src_new_height;
    roi.width = src_new_width;
    roi.x = 0;
    roi.y = 0;
//    roi.x = (des_width - src_new_width) / 2;
//    roi.y = (des_height - src_new_height) / 2;

    src_resize.copyTo(des(roi));
}
