//
// Created by lizhaoliang-os on 2021/3/4.
//

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/photo/photo.hpp"

#include "Module_yolov5.h"
#include "face_640_small_int8_bgr.h"

std::vector<std::string> split(const std::string& string, char separator, bool ignore_empty) {
    std::vector<std::string> pieces;
    std::stringstream ss(string);
    std::string item;
    while (getline(ss, item, separator)) {
        if (!ignore_empty || !item.empty()) {
            pieces.push_back(std::move(item));
        }
    }
    return pieces;
}

std::string trim(const std::string& str) {
    size_t left = str.find_first_not_of(' ');
    if (left == std::string::npos) {
        return str;
    }
    size_t right = str.find_last_not_of(' ');
    return str.substr(left, (right - left + 1));
}

int main(int argc, char* argv[])
{
    std::string deploy_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    input_names.clear();
    input_names.emplace_back("input");

    output_names.clear();
    output_names.emplace_back("output1");
    output_names.emplace_back("output2");
    output_names.emplace_back("output3");

    YoloConfig config_tmp;
    float means_rgb[3] = {0, 0, 0};
    float scales_rgb[3] = {0.0039215, 0.0039215, 0.0039215}; // 1.0 / 255

    config_tmp.means[0] = means_rgb[0];
    config_tmp.means[1] = means_rgb[1];
    config_tmp.means[2] = means_rgb[2];

    config_tmp.scales[0] = scales_rgb[0];
    config_tmp.scales[1] = scales_rgb[1];
    config_tmp.scales[2] = scales_rgb[2];

    config_tmp.mean_length = 3;
    config_tmp.net_inp_channels = 3;

    config_tmp.num_threads = 4;

    config_tmp.conf_thres = 0.4;
    config_tmp.nms_thresh = 0.5;

    config_tmp.strides = {8, 16, 32};
    config_tmp.anchor_grids = { {10, 13, 16, 30, 33, 23} , {30, 61, 62, 45, 59, 119}, {116, 90, 156, 198, 373, 326} };

    std::string project_root = "./";
    if(argc < 4)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << "input_size image_list save_root"
                  << std::endl;
        return -1;
    }

    int input_size = std::atoi(argv[1]);

    /*******************Yolov5******************/
    CModule_yolov5 yolov5;
    int model_size;
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.model_buffer = const_cast<uint8_t*>(face_640_small_int8_bgr_rknn);
    config_tmp.model_size = face_640_small_int8_bgr_rknn_len;
    config_tmp.net_inp_width = input_size;
    config_tmp.net_inp_height = config_tmp.net_inp_width;
    yolov5.init(config_tmp);

    std::string win_track = "Tracking";
    std::ifstream input(argv[2]);
    std::string line;

    std::string img_path;
    std::string img_save_root = std::string(argv[3]);
    long frame_id = 0;
    while (true)
    {
        cv::Mat frame;
        std::getline(input, line);
        if ("" == line)
            break;
        img_path = trim(line);
        frame = cv::imread(img_path);
        std::vector<std::string> items = split(line, '/', true);
        if (!frame.data)
        {
            break;
        }

        cv::Mat img_origin = frame.clone();
        cv::Mat img_show = frame.clone();

        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        yolov5.process(&frame);
        const std::vector<BoxInfo>& det = yolov5.get_result();
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();
        std::cout << "frame_id:" << frame_id << " Using rv1126 all time = " << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP1 - startTP).count() << " ms" << std::endl;
        std::cout << "Detected obj num : " << det.size() << std::endl;
        frame_id++;
        // show result
        std::string img_save_path = img_save_root + "/" + items[items.size() - 1];
        for (size_t idx = 0; idx < det.size(); idx++)
        {
            int xmin = det[idx].x1;
            int ymin = det[idx].y1;
            int xmax = det[idx].x2;
            int ymax = det[idx].y2;
            float score = det[idx].score;
            int label = det[idx].label;
            img_save_path += "_xywh_" + std::to_string(xmin) + "_" + std::to_string(ymin) + "_" +
                             std::to_string(xmax - xmin) + "_" + std::to_string(ymax - ymin);
            cv::rectangle(img_show, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255, 0, 0), 2);
            cv::putText(img_show, std::to_string(score), cv::Point(xmax, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
        }
        img_save_path += ".jpg";
        std::cout << "Save img to " << img_save_path << std::endl;
        cv::imwrite(img_save_path, img_show);
    }
    return 0;
}
