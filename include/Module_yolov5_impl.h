#ifndef MODULE_YOLOV5_IMPL_H
#define MODULE_YOLOV5_IMPL_H

#include <string>
#include <vector>

#include "data_type.h"

class CModule_yolov5_impl
{
public:
	CModule_yolov5_impl();
	virtual ~CModule_yolov5_impl() ;

    void init(const YoloConfig &config);

    void process(const void* mat);

	const std::vector<BoxInfo>& get_result();

protected:
    virtual void engine_init() = 0;
    virtual void engine_run() = 0;
    virtual void engine_post(const std::vector<std::string>& output_names) = 0;
	virtual void preprocess();

private:
    void yolo_resize_with_opencv(const void* mat_ptr, int des_height, int des_width);

protected:
    YoloConfig config_;
    std::vector<float> data_out_;
	std::vector<BoxInfo> boxs_;
    int img_height_;
    int img_width_;
    int step_each_obj_;
    void* des_mat_;
    uint8_t* src_resize_ptr_;
};

#endif // MODULE_YOLOV5_IMPL_H

