# ONNX模型RK1126平台转换量化(以yolov5为例)


## 环境配置

- 参考[RKNN_Toolkit_V1.7.1的官方文档](doc/Rockchip_User_Guide_RKNN_Toolkit_V1.7.1_CN.pdf)配置，建议使用官方提供的docker


## 模型转换

- ONNX模型来源见[yolov5_train_and_convert#6转换为onnx模型](https://github.com/jnulzl/yolov5_train_and_convert#6%E8%BD%AC%E6%8D%A2%E4%B8%BAonnx%E6%A8%A1%E5%9E%8B)，这里必须要用*多输出版本*的ONNX模型

- 验证ONNX模型

```shell
python demo_onnx.py models/face_640_small.onnx 640  img_list.txt
```

- 量化ONNX模型


```shell
python detect_int8.py models/face_640_small.onnx 640  # int8量化
python detect_int16.py models/face_640_small.onnx 640 # int16量化
```

**注意：**：

1.*这里使用int8量化，且reorder_channel='2 1 0'，此时，当调用量化后的模型时，无需再进行rgb2bgr或rgb2bgr操作，这个要特别注意，具体可参考demo_rknn.py中的预处理操作*

2.*对于那些对精度要求高的模型，例如：人脸/人体关键点，建议使用int16量化*

- 验证量化后的模型(RKNN)

```shell
python demo_rknn.py models/face_640_small_int8_bgr.rknn 640  img_list.txt
```
