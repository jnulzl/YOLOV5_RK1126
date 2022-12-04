//
// Created by lizhaoliang-os on 2020/6/9.
//

#ifndef MODULE_YOLOV5_RV1126_IMPL_H
#define MODULE_YOLOV5_RV1126_IMPL_H

#include "Module_yolov5_impl.h"
#include "rknn_api.h"

class CModule_yolov5_rv1126_impl : public CModule_yolov5_impl
{
public:
    CModule_yolov5_rv1126_impl();
    virtual ~CModule_yolov5_rv1126_impl();

private:
    virtual void engine_init() override;
    virtual void engine_run() override;
    virtual void engine_post(const std::vector<std::string>& output_names) override;
    virtual void preprocess() override;

private:
    rknn_context ctx_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_output> outputs_;
};

#endif //MODULE_YOLOV5_RV1126_IMPL_H
