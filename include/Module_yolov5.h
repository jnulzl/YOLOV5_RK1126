#ifndef MODULE_YOLOV5_H
#define MODULE_YOLOV5_H

#include <string>
#include <vector>
#include "data_type.h"
#include "alg_define.h"

class ALG_PUBLIC CModule_yolov5
{
public:
	CModule_yolov5();

	~CModule_yolov5();

	void init(const YoloConfig& config);

    void process(const void* mat);

    /**
     * 该接口仅适用于非合并模型
     * @return
     */
    const std::vector<BoxInfo>& get_result();

private:
	AW_ANY_POINTER impl_;
};

#endif // MODULE_YOLOV5_H

