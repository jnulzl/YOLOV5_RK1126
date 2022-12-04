# ONNX模型RK1126平台部署(以yolov5为例)


## 环境配置

- Ubuntu18.04/Ubuntu20.04

- cmake, git, make等等,根据实际情况缺什么安装什么

- 交叉编译工具[gcc-arm-9.2-2019.12-x86_64-arm-none-linux-gnueabihf](https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-arm-none-linux-gnueabihf.tar.xz?revision=fed31ee5-2ed7-40c8-9e0e-474299a3c4ac&rev=fed31ee52ed740c89e0e474299a3c4ac&hash=BD8F056BFC89F1C62F1C6D0786B7769126CF325E)

## 模型量化

[见models/detect_yolov5_onnx/README.md](models/detect_yolov5_onnx/README.md)

## 构建

```shell
cd $ROOT
mkdir build && cd build
export CROSS_COMPILER_ROOT = 'YOUR CROSS COMPILER ROOT'
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$CROSS_COMPILER_ROOT/bin/arm-none-linux-gnueabihf-g++ -DCMAKE_C_COMPILER=$CROSS_COMPILER_ROOT/bin/arm-none-linux-gnueabihf-gcc
```
输出产物在`$ROOT/bin/Linux`下面，拷贝到板子跑即可。

## 將模型转化为buffer直接编译到库或者可执行文件中

[见examples/README.md](examples/README.md)

