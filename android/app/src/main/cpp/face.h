#ifndef JNITEST_IMAGE_FACE_H
#define JNITEST_IMAGE_FACE_H
#include"Define.h"
class CImage_face
{
public:
    CImage_face();
    ~CImage_face();
    //敏感和热力图
    string StringAdd(string Path, string addPath);

private:

};



#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "wenfen", ##__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "wenfen", ##__VA_ARGS__);
#endif //JNITEST_IMAGE_ENGINE_H
