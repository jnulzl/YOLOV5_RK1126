import os
import sys
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

pre_compile = False

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


def postprocess(output):
    print("output : ", output.shape)
    # reshape + permuate + Sigmoid
    shape = output.shape
    data = output.reshape(3, 6, shape[2], shape[3])
    data = data.transpose(0, 2, 3, 1)
    data = sigmoid(data)

    return data
     

if __name__ == '__main__':

    if 3 != len(sys.argv):
        print("Usage:\n\t %s onnx_path input_size"%(sys.argv[0]))
        sys.exit(0)
    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> Config model')
    rknn.config(batch_size=1, mean_values=[[0.0, 0.0, 0.0]], std_values=[[255, 255, 255]], 
                # asymmetric_quantized-u8 dynamic_fixed_point-8 dynamic_fixed_point-16
                quantized_dtype = 'asymmetric_quantized-u8',
                reorder_channel='2 1 0',  # '2 1 0': swap b r channel and inference, '0 1 2':inference
                # optimization_level=3, 
                # output_optimize=1, 
                # quantize_input_node=True,
                target_platform='rv1126')
    # print(help(rknn.config))    
    print('done')
    deploy = sys.argv[1]
    input_size = int(sys.argv[2])
    # Load caffe model    
    print('--> Loading model')    
    # ret = rknn.load_caffe(model=deploy, proto='caffe', blobs=caffemodel)
    ret = rknn.load_onnx(model=deploy)
    if ret != 0:
        print('Load interp_test failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

    # Build model
    if pre_compile:
        rknn_model = deploy.replace('.onnx', '_int8_pre_compile.rknn')
    else:
        rknn_model = deploy.replace('.onnx', '_int8_bgr.rknn')

    print('--> Building model')
    size_data_map = {
                     320:'./dataset_320.txt',
                     640:'./dataset_640.txt'
                    }
    ret = rknn.build(do_quantization=True, dataset=size_data_map[input_size], pre_compile=pre_compile)
    # print(help(rknn.build))
    # exit(0)

    if ret != 0:
        print('Build onnx model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        print('Export onnx model.rknn failed!')
        exit(ret)
    print('done')

    rknn.release()

