#include "Module_yolov5_impl.h"
#include "Module_yolov5.h"

#include "Module_yolov5_rv1126_impl.h"
#include "debug.h"

CModule_yolov5::CModule_yolov5()
{
    impl_ = new CModule_yolov5_rv1126_impl();
}

CModule_yolov5::~CModule_yolov5()
{
    delete ANY_POINTER_CAST(impl_, CModule_yolov5_impl);
#if defined(ALG_DEBUG) || defined(ALPHAPOSE_DEBUG)
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_yolov5::init(const YoloConfig& config)
{
    ANY_POINTER_CAST(impl_, CModule_yolov5_impl)->init(config);
}

void CModule_yolov5::process(const void* mat)
{
    ANY_POINTER_CAST(impl_, CModule_yolov5_impl)->process(mat);
}

const std::vector<BoxInfo>& CModule_yolov5::get_result()
{
    return ANY_POINTER_CAST(impl_, CModule_yolov5_impl)->get_result();
}
