import os
import sys
import time
import numpy as np
import torch
import torchvision
import cv2
import onnxruntime as ort

BOX_THRESH = 0.5
NMS_THRESH = 0.6

strides = [8, 16, 32]
anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

CLASSES = ('Face')


def preprocess(img, new_shape):
    height, width = img.shape[:2]
    scale_wh = new_shape / max(height, width)
    new_height, new_width = int(scale_wh * height), int(scale_wh * width)
    img_resize = cv2.resize(img, (new_width, new_height))

    image = np.zeros((new_shape, new_shape, 3), np.uint8)
    if new_height >= new_width:
        xmin, ymin = (new_height - new_width) // 2, 0
    if new_height <= new_width:
        xmin, ymin = 0, (new_width - new_height) // 2
    image[ymin:ymin + new_height, xmin:xmin + new_width, :] = img_resize

    return image

def sigmoid(data):
    return 1 / (1 + np.exp(-1 * data))
    

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """    
    nc = prediction.shape[1] - 5  # number of classes
    xc = prediction[:, 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    t = time.time()
    x = prediction  # 6300x(5+cls_num)
    x = x[xc]  # confidence

    # If none remain process next image
    if not x.shape[0]:
        return None

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)    
    conf, j = x[:, 5:].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
    
    # If none remain process next image
    if not x.shape[0]:
        return None
        
    boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]

    return x[i].numpy()

def make_grid(nx=4, ny=3):
    data = np.zeros(2 * nx * ny, dtype=np.float)
    for y in range(ny):
        for x in range(nx):
            data[2 * x + 2 * nx * y] = x
            data[2 * x + 2 * nx * y + 1] = y
    data = data.reshape(1, ny, nx, 2)
    return data.flatten()


def postprocess_help(src, src_n, src_c, src_h, src_w, stride, anchor_grid):
    '''
        src         : float*
        stride      : float
        anchor_grid : float*
    '''

    ''' n  c  h w
        3x40x40x6
        3x40x40x[:2] - 1x40x40x2
    '''
    # y[:, :, :, 0:2] = (y[:, :, :, 0:2] * 2. - 0.5 + make_grid_me(ny, nx)) * stride_tmp  # xy
    '''
        3x40x40x[2:4] * 3x1x1x2
    '''
    # y[:, :, :, 2:4] = (y[:, :, :, 2:4] * 2) ** 2 * anchor_grid_tmp  # wh
    chw = src_c * src_h * src_w
    ch = src_c * src_h
    grid = make_grid(nx=src_h, ny=src_c)
    for bs in range(src_n):
        for index in range(ch):
            step = bs * chw + index * src_w
            x = src[step + 0]
            y = src[step + 1]
            w = src[step + 2]
            h = src[step + 3]
            src[step + 0] = (x * 2 - 0.5 + grid[index * 2 + 0]) * stride  # x
            src[step + 1] = (y * 2 - 0.5 + grid[index * 2 + 1]) * stride  # y
            # wh
            src[step + 2] = w * 2 * w * 2 * anchor_grid[2 * bs + 0]  # w
            src[step + 3] = h * 2 * h * 2 * anchor_grid[2 * bs + 1]  # h

    return src

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    '''
        img1_shape : H,W
        img0_shape : H,W,C
    '''
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def yolov5_post_process(z, net_input, img_origin):
    '''
        z       : np.array, size = [[3, 40, 40, 6],[3, 20, 20, 6],[3, 10, 10, 6]]
        strides : list, 1x3, data = [8, 16, 32] 
        anchors : list, 3x6, data = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
    '''
    res = []
    num = len(z)
    for index in range(num):
        n, c, h, w = np.array(z[index].shape).astype(np.int).tolist()
        res_tmp = postprocess_help(z[index].flatten().tolist(), n, c, h, w, strides[index], anchors[index])
        res += res_tmp
    data = np.array(res).astype(np.float32)
    pred = torch.from_numpy(data.reshape(-1, w))
    det = non_max_suppression(pred, BOX_THRESH, NMS_THRESH)
    
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        # print(net_input, det.shape, img_origin.shape)
        det[:, :4] = scale_coords(net_input.shape[2:], det[:, :4], img_origin.shape).round()
    return det
    

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.
    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
                    

def postprocess(output):
    print("output : ", output.shape)
    # reshape + permuate + Sigmoid
    shape = output.shape
    data = output.reshape(3, -1, shape[2], shape[3])
    data = data.transpose(0, 2, 3, 1)
    data = sigmoid(data)

    return data
     

if __name__ == '__main__':
    
    if 4 != len(sys.argv):
        print("Usage:\n\t %s onnx_path input_size img_list"%(sys.argv[0]))
        exit(0)
    
    # 1. Load onnx model
    onnx_path = sys.argv[1]
    input_size = int(sys.argv[2])
    model_ort = ort.InferenceSession(onnx_path)
       
    
    # 2. Read img and preprocess
    if not os.path.exists("res"):
        os.makedirs("res")
    with open(sys.argv[3], 'r') as fpR:
        lines = fpR.readlines()
        for line in lines:
            img_path = line.strip()
            img = cv2.imread(img_path)
            img_show = img.copy()
            img = preprocess(img, input_size).astype(np.float32) # resize
            img = img[:, :, ::-1] # BGR -> RGB
            mean = [0, 0, 0]
            std = [255, 255, 255]
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)
            img = img.reshape(1, *img.shape).astype(np.float32)

            # 3. Inference
            # print("ONNX Input shape: ", img.shape)
            loop = 10
            begin = time.time()
            for _ in range(loop):
                outputs = model_ort.run(['output1', 'output2', 'output3'], {'input': img})
            print('Average inference time. %dms' %(1000 * (time.time() - begin) / loop))
            
            # for output in outputs:
            #     print("Output shape : ", output.shape)
                
            det = yolov5_post_process(outputs, img, img_show)
            if det is None:
                continue
            boxes = det[:,0:4]
            scores = det[:,4]
            classes = det[:,5].astype(np.int32)
            
            if boxes is not None:
                draw(img_show, boxes, scores, classes)
            cv2.imwrite("res/" + os.path.basename(img_path), img_show)
