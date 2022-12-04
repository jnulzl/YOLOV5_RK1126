//
// Created by jnulzl on 2020/6/20.
//
#include <vector>
#include <algorithm>
#include "post_process.h"

std::vector<int> make_grid(int nx, int ny) {
    std::vector<int> data;
    long num = 2 * nx * ny;
    data.resize(num);
    for (int  idy = 0; idy < ny; idy++)
    {
        for (int idx = 0; idx < nx; idx++)
        {
            long index = 2 * idx + 2 * nx * idy;
            data[index] = idx;
            data[index + 1] = idy;
        }
    }

    return data;
}

void decode_net_output(float* src, int src_n, int src_c, int src_h, int src_w,
                       int stride, const float* anchor_grid) {
    int chw = src_c * src_h * src_w;
    int ch = src_c * src_h;
    std::vector<int> grid = make_grid(src_h, src_c);
    for (long bs = 0; bs < src_n; bs++)
    {
        for (long index = 0; index < ch; index++)
        {
            long step = bs * chw + index * src_w;
            float x = src[step + 0];
            float y = src[step + 1];
            float w = src[step + 2];
            float h = src[step + 3];
            // xy
            src[step + 0] = (x * 2 - 0.5 + grid[index * 2 + 0]) * stride;  // x
            src[step + 1] = (y * 2 - 0.5 + grid[index * 2 + 1]) * stride;  // y
            // wh
            src[step + 2] = w * 2 * w * 2 * anchor_grid[2 * bs + 0];  // w
            src[step + 3] = h * 2 * h * 2 * anchor_grid[2 * bs + 1];  // h
        }
    }
}

void nms(std::vector<BoxInfo>& input_boxes, float nms_thresh) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) {return a.score > b.score; });
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_thresh)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

void non_max_suppression(float* src, int src_height, int src_width, float conf_thres, float nms_thresh,
                         std::vector<BoxInfo>& dets) {
    dets.clear();

    int num_cls = src_width - 5;
    for (long idx = 0; idx < src_height; idx++) {
        const int step = idx * src_width;
        /* cx, cy, width, height, obj_conf, cls_conf0, cls_conf1, ..., cls_conf{n} */
        float obj_conf = src[step + 4];
        if (obj_conf < conf_thres)
            continue;
        float max_score = -1.0f;
        int   obj_cls = -1;
        for (int idy = 5; idy < src_width; idy++) {
            float tmp = src[step + idy] * obj_conf;
            if (tmp > max_score) {
                max_score = tmp;
                obj_cls = idy - 5;
            }
        }
        if (max_score < conf_thres)
            continue;
        /* xywh2xyxy */
        BoxInfo box;
        float cx = src[step + 0];
        float cy = src[step + 1];
        float width_2 = src[step + 2] / 2;
        float height_2 = src[step + 3] / 2;
        box.x1 = cx - width_2;
        box.y1 = cy - height_2;
        box.x2 = cx + width_2;
        box.y2 = cy + height_2;
        box.score = max_score;
        box.label = obj_cls;
        dets.push_back(box);
    }


    nms(dets, nms_thresh);
}


void scale_coords(BoxInfo& box, float net_input_height, float net_input_width,
                  float img_height, float img_width) {

#if 0
    float gain_x = 1.0f * net_input_width / img_width;
	float gain_y = 1.0f * net_input_height / img_height;
	float pad_w = 0.0f;
	float pad_h = 0.0f;
	//float pad_w = (net_input_width - img_width * gain_x) / 2.0f;
	//float pad_h = (net_input_height - img_height * gain_y) / 2.0f;

#else
    float gain_x = 1.0f * std::max(net_input_height, net_input_width) / std::max(img_height, img_width);
    float gain_y = gain_x;
    float pad_w = 0.0f;
    float pad_h = 0.0f;
    //float pad_w = (net_input_width - img_width * gain_x) / 2.0f;
    //float pad_h = (net_input_height - img_height * gain_y) / 2.0f;
#endif


    box.x1 = (box.x1 - pad_w) / gain_x;
    box.x2 = (box.x2 - pad_w) / gain_x;
    box.y1 = (box.y1 - pad_h) / gain_y;
    box.y2 = (box.y2 - pad_h) / gain_y;

    box.x1 = box.x1 > 0 ? box.x1 : 0;
    box.y1 = box.y1 > 0 ? box.y1 : 0;
    box.x2 = box.x2 < img_width ? box.x2 : img_width - 1;
    box.y2 = box.y2 < img_height ? box.y2 : img_height - 1;
}


void postprocess(std::vector<BoxInfo>& dets, float net_input_height, float net_input_width,
                 float img_height, float img_width) {
    long dets_num = dets.size();
    for (long idx = 0; idx < dets_num; idx++)
    {
        scale_coords(dets[idx], net_input_height, net_input_width, img_height, img_width);
    }
}

