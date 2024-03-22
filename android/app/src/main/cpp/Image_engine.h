//
// Created by Administrator on 2023/7/19.
//

#ifndef JNITEST_IMAGE_ENGINE_H
#define JNITEST_IMAGE_ENGINE_H
#include"Define.h"
class CImage_engine
{
public:
    CImage_engine();
    ~CImage_engine();
      //敏感和热力图
    string StringAdd(string Path, string addPath);
    
private:

};



#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "wenfen", ##__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "rkyolo4j", ##__VA_ARGS__);
#endif //JNITEST_IMAGE_ENGINE_H
