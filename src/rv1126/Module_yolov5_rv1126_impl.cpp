//
// Created by lizhaoliang-os on 2020/6/9.
//

#include "opencv2/opencv.hpp"
#include "Module_yolov5_rv1126_impl.h"
#include "post_process.h"
#include <unistd.h>
#include "debug.h"

#ifdef ALG_DEBUG
static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}
#endif

CModule_yolov5_rv1126_impl::CModule_yolov5_rv1126_impl()
{

}

CModule_yolov5_rv1126_impl::~CModule_yolov5_rv1126_impl()
{
    // Release
    if (ctx_ >= 0)
    {
        rknn_destroy(ctx_);
    }
#ifdef ALG_DEBUG
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_yolov5_rv1126_impl::preprocess()
{

}

void CModule_yolov5_rv1126_impl::engine_init()
{
    int ret = rknn_init(&ctx_, config_.model_buffer, config_.model_size, 0);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        exit(1);
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        exit(1);
    }
#ifdef ALG_DEBUG
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            exit(1);
        }
        printRKNNTensor(&(input_attrs[i]));
    }
    printf("output tensors:\n");
#endif
    output_attrs_.resize(io_num.n_output);
    memset(output_attrs_.data(), 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            exit(1);
        }
#ifdef ALG_DEBUG
        printRKNNTensor(&(output_attrs_[i]));
#endif
    }
}

void CModule_yolov5_rv1126_impl::engine_run()
{
#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif

    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = sizeof(uint8_t) * config_.net_inp_channels * config_.net_inp_height * config_.net_inp_width;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = src_resize_ptr_;

#ifdef ALG_DEBUG
    cv::Mat src_tmp(config_.net_inp_height, config_.net_inp_width, CV_8UC3, src_mat_.data());
    cv::imwrite("src_tmp.jpg", src_tmp);
#endif

    int ret = rknn_inputs_set(ctx_, 1, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        exit(1);
    }

    // Run
    ret = rknn_run(ctx_, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        exit(1);
    }

#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    std::printf("rv1126 inference time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count());
#endif

    // Get Output
    int net_output_num = config_.output_names.size();
    if(outputs_.empty())
    {
        outputs_.resize(net_output_num);
        for (int idx = 0; idx < net_output_num; ++idx)
        {
            outputs_[idx].want_float = 1;
        }
    }
    ret = rknn_outputs_get(ctx_, net_output_num, outputs_.data(), NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        exit(1);
    }

    if(data_out_.empty())
    {
        int output_num = 0;
        for (size_t idx = 0; idx < net_output_num; idx++)
        {
            output_num += output_attrs_[idx].n_elems;
        }
        data_out_.resize(output_num);
    }
#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
    std::printf("postprocess0 time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(begin_time_nms - end_time).count());
#endif

    // post process
    float* data_ = data_out_.data();
    int step_tmp = 0;
    step_each_obj_ = output_attrs_[0].dims[0]; // 5 + cls_num
    for (size_t idx = 0; idx < net_output_num; idx++)
    {
#ifdef ALG_DEBUGa
        std::cout << "AAAAAAAAA : " << 3 << " " << output_attrs_[idx].dims[2] / 3 << " "
                  << output_attrs_[idx].dims[1] << " " << output_attrs_[idx].dims[0] << std::endl;
        std::cout << "BBBBBBBBB : " << output_shape[0] << " "
                  << output_shape[1] << " " << output_shape[2] << " " << output_shape[3] << std::endl;
#endif
        memcpy(data_ + step_tmp,  outputs_[idx].buf, outputs_[idx].size);
        if(1 != net_output_num)
        {
            decode_net_output(data_ + step_tmp,
                              output_attrs_[idx].dims[3], output_attrs_[idx].dims[2],
                              output_attrs_[idx].dims[1], output_attrs_[idx].dims[0],
                              config_.strides[idx], config_.anchor_grids[idx].data());
        }
        step_tmp += outputs_[idx].size / sizeof(float);
    }
#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_nms = std::chrono::system_clock::now();
    std::printf("postprocess1 time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count());
#endif
    rknn_outputs_release(ctx_, outputs_.size(), outputs_.data());
}

void CModule_yolov5_rv1126_impl::engine_post(const std::vector<std::string> &output_names)
{
    if(config_.output_names.size() != output_names.size())
    {
        config_.output_names = output_names;
        engine_run();
    }
}