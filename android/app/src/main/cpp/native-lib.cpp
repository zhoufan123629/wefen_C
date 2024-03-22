#include <jni.h>
#include <string>
#include"code_enum.h"
#include<vector>
code_enum code_enum_;
#include<opencv2/opencv.hpp>
#include"Image_engine.h"
#include"face.h"

#define TAG "wefen"
#define LogI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LogD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LogE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)


//region 模型
#include"yolo.h"
#include "yolo_ncnn.h"
#include "config.h"
static ncnn::Mutex Lock;
//yoloNcnn 目标检测 yolo 分割
Yolo *g_yolo_follicle= nullptr;
Yolo *g_yolo_segmentation= nullptr;
Yolo *g_yolo_scurf= nullptr;
Yolo *g_yolo_follicle_inflammation= nullptr;
YoloNcnn *g_yoloNcnn_whitehair= nullptr;
YoloNcnn *g_yoloNcnn_thickness=nullptr;
YoloNcnn *g_yoloNcnn_density=nullptr;
YoloNcnn *g_yoloncnn_spot= nullptr;
//endregion 模型
std::string StringAdd(std::string Path, std::string addPath) {
    Path.append(addPath);
    return Path;
};

//region 头皮检测

//region 1.头皮油脂
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1scalpOil(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                jstring dst_img_path, jintArray display_img,
                                                jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_scalpOil()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>oil_areas;
    std::vector<int>oil_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(oil_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, oil_areas.size(), oil_areas.data());
    jintArray class_num_array = env->NewIntArray(oil_class.size());
    env->SetIntArrayRegion(class_num_array, 0, oil_class.size(), oil_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_scalpoil"));
    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    oil_areas.clear();
    oil_class.clear();
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 1.头皮油脂

//region 2.油脂堵塞
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initOilBlockageModel(JNIEnv *env, jclass clazz, jstring model_path_bin,
                                                 jstring model_path_param) {
    // TODO: implement initOilBlockageModel()
    const float mean_vals[] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int target_size = 640;
    config config_;
    int result = config_.model_init_result_succ;
    // reload
    const char *param_path = env->GetStringUTFChars( model_path_param, NULL);
    const char *bin_path = env->GetStringUTFChars( model_path_bin, NULL);
    ncnn::MutexLockGuard g(Lock);
    if (!g_yolo_follicle) {
        g_yolo_follicle = new Yolo;
        result = g_yolo_follicle->follicle_load( target_size, mean_vals, norm_vals,param_path,bin_path);
    }
    env->ReleaseStringUTFChars( model_path_bin, bin_path);
    env->ReleaseStringUTFChars( model_path_param, param_path);
    return result;
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1oilBlockage(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                   jstring dst_img_path, jintArray display_img,
                                                   jobjectArray dst_img_name, jobject entity) {
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<Object>objects;
    g_yolo_follicle->follicle_detect(useImage,objects,0.1f,0.4f,1);
    g_yolo_follicle->follicle_draw(useImage, objects);
    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>oil_blockage_areas;
    std::vector<int>oil_blockage_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(oil_blockage_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, oil_blockage_areas.size(), oil_blockage_areas.data());
    jintArray class_num_array = env->NewIntArray(oil_blockage_class.size());
    env->SetIntArrayRegion(class_num_array, 0, oil_blockage_class.size(), oil_blockage_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_oilblockage"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    oil_blockage_areas.clear();
    oil_blockage_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 2.油脂堵塞

//region 3.斑点
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initSpotModel(JNIEnv *env, jclass clazz, jstring model_path_bin,
                                          jstring model_path_param) {
    // TODO: implement initSpotModel()
    const float mean_vals[] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int target_size = 640;
    config config_;
    int result = config_.model_init_result_succ;
    // reload
    const char *param_path = env->GetStringUTFChars( model_path_param, NULL);
    const char *bin_path = env->GetStringUTFChars( model_path_bin, NULL);
    ncnn::MutexLockGuard g(Lock);
    if (!g_yoloncnn_spot) {
        g_yoloncnn_spot = new YoloNcnn;
        result = g_yoloncnn_spot->spot_load( target_size, mean_vals, norm_vals,param_path,bin_path);
    }
    env->ReleaseStringUTFChars( model_path_bin, bin_path);
    env->ReleaseStringUTFChars( model_path_param, param_path);
    return result;

    
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1spot(JNIEnv *env, jclass clazz, jstring src_img_path,
                                            jstring dst_img_path, jintArray display_img,
                                            jobjectArray dst_img_name, jobject entity) {
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<target_detection_Object>objects;
    g_yoloncnn_spot->spot_detect(useImage,objects,0.1f,0.4f,1);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>spot_areas;
    std::vector<int>spot_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(spot_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, spot_areas.size(), spot_areas.data());
    jintArray class_num_array = env->NewIntArray(spot_class.size());
    env->SetIntArrayRegion(class_num_array, 0, spot_class.size(), spot_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_spot"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    spot_areas.clear();
    spot_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 3.斑点

//region 4.头皮敏感
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initSegmentationModel(JNIEnv *env, jclass clazz, jstring model_path_bin,
                                                  jstring model_path_param) {
    // TODO: implement initSegmentationModel()
    const float mean_vals[] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int target_size = 640;
    config config_;
    int result = config_.model_init_result_succ;
    // reload
    const char *param_path = env->GetStringUTFChars( model_path_param, NULL);
    const char *bin_path = env->GetStringUTFChars( model_path_bin, NULL);
    ncnn::MutexLockGuard g(Lock);
    if (!g_yolo_segmentation) {
        g_yolo_segmentation = new Yolo;
        result = g_yolo_segmentation->hair_segmentation_load_new( target_size, mean_vals, norm_vals,param_path,bin_path);
    }
    env->ReleaseStringUTFChars( model_path_bin, bin_path);
    env->ReleaseStringUTFChars( model_path_param, param_path);
    return result;
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1scalpsensitivity(JNIEnv *env, jclass clazz,
                                                        jstring src_img_path, jstring dst_img_path,
                                                        jintArray display_img,
                                                        jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_scalpsensitivity()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<Object>objects;
    g_yolo_segmentation->hair_segmentation_detect_new(useImage,objects,0.07f,0.9f,1);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>sensitivity_areas;
    std::vector<int>sensitivity_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(sensitivity_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, sensitivity_areas.size(), sensitivity_areas.data());
    jintArray class_num_array = env->NewIntArray(sensitivity_class.size());
    env->SetIntArrayRegion(class_num_array, 0, sensitivity_class.size(), sensitivity_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_sensitivity"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    sensitivity_areas.clear();
    sensitivity_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 4.头皮敏感

//region 5.水油平衡
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1wateroil(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                jstring dst_img_path, jintArray display_img,
                                                jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_wateroil()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();


    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>wateroil_areas;
    std::vector<int>wateroil_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(wateroil_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, wateroil_areas.size(), wateroil_areas.data());
    jintArray class_num_array = env->NewIntArray(wateroil_class.size());
    env->SetIntArrayRegion(class_num_array, 0, wateroil_class.size(), wateroil_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_wateroil"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    wateroil_areas.clear();
    wateroil_class.clear();
    useImage.release();

    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 5.水油平衡

//region 6.角质头屑
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initScurfModel(JNIEnv *env, jclass clazz, jstring model_path_bin,
                                           jstring model_path_param) {
    // TODO: implement initScurfModel()
    const float mean_vals[] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int target_size = 640;
    config config_;
    int result = config_.model_init_result_succ;
    // reload
    const char *param_path = env->GetStringUTFChars( model_path_param, NULL);
    const char *bin_path = env->GetStringUTFChars( model_path_bin, NULL);
    ncnn::MutexLockGuard g(Lock);
    if (!g_yolo_scurf) {
        g_yolo_scurf = new Yolo;
        result = g_yolo_scurf->hair_cutin_load( target_size, mean_vals, norm_vals,param_path,bin_path);
    }
    env->ReleaseStringUTFChars( model_path_bin, bin_path);
    env->ReleaseStringUTFChars( model_path_param, param_path);
    return result;
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1cutinscurf(JNIEnv *env, jclass clazz, jstring src_img_path,
                                             jstring dst_img_path, jintArray display_img,
                                             jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_scurf()

    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<Object>objects;
    g_yolo_scurf->hair_cutin_detect(useImage,objects,0.1f, 0.4f, 1);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>scurf_areas;
    std::vector<int>scurf_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(scurf_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, scurf_areas.size(), scurf_areas.data());
    jintArray class_num_array = env->NewIntArray(scurf_class.size());
    env->SetIntArrayRegion(class_num_array, 0, scurf_class.size(), scurf_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_cutinscurf"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    scurf_areas.clear();
    scurf_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;

}
//endregion 6.角质头屑

//region 7.粉状头屑
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1powderyscurf(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                    jstring dst_img_path, jintArray display_img,
                                                    jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_powderyscurf()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<Object>objects;
    g_yolo_scurf->hair_cutin_detect(useImage,objects,0.1f, 0.4f, 1);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>powderyscurf_areas;
    std::vector<int>powderyscurf_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(powderyscurf_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, powderyscurf_areas.size(), powderyscurf_areas.data());
    jintArray class_num_array = env->NewIntArray(powderyscurf_class.size());
    env->SetIntArrayRegion(class_num_array, 0, powderyscurf_class.size(), powderyscurf_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_powderyscurf"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    powderyscurf_areas.clear();
    powderyscurf_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 7.粉状头屑

//region 8.头发稀疏
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initDensityModel(JNIEnv *env, jclass clazz, jstring model_path_bin,
                                             jstring model_path_param) {
    // TODO: implement initDensityModel()
    const float mean_vals[] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int target_size = 640;
    config config_;
    int result = config_.model_init_result_succ;
    // reload
    const char *param_path = env->GetStringUTFChars( model_path_param, NULL);
    const char *bin_path = env->GetStringUTFChars( model_path_bin, NULL);
    ncnn::MutexLockGuard g(Lock);
    if (!g_yoloNcnn_density) {
        g_yoloNcnn_density = new YoloNcnn;
        result = g_yoloNcnn_density->load( target_size, mean_vals, norm_vals,param_path,bin_path);
    }
    env->ReleaseStringUTFChars( model_path_bin, bin_path);
    env->ReleaseStringUTFChars( model_path_param, param_path);
    return result;
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1hairsparse(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                   jstring dst_img_path, jintArray display_img,
                                                   jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_scalpsparse()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<target_detection_Object>objects;
    g_yoloNcnn_density->yolov8_detect(useImage,objects,0.1f, 0.06f, 3);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>scalpsparse_areas;
    std::vector<int>scalpsparse_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(scalpsparse_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, scalpsparse_areas.size(), scalpsparse_areas.data());
    jintArray class_num_array = env->NewIntArray(scalpsparse_class.size());
    env->SetIntArrayRegion(class_num_array, 0, scalpsparse_class.size(), scalpsparse_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_hairsparse"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    scalpsparse_areas.clear();
    scalpsparse_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 8.头发稀疏

//region 9.头发细软
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initThicknessModel(JNIEnv *env, jclass clazz, jstring model_path_bin,
                                               jstring model_path_param) {
    // TODO: implement initThicknessModel()
    const float mean_vals[] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int target_size = 640;
    config config_;
    int result = config_.model_init_result_succ;
    // reload
    const char *param_path = env->GetStringUTFChars( model_path_param, NULL);
    const char *bin_path = env->GetStringUTFChars( model_path_bin, NULL);
    ncnn::MutexLockGuard g(Lock);
    if (!g_yoloNcnn_thickness) {
        g_yoloNcnn_thickness = new YoloNcnn;
        result = g_yoloNcnn_thickness->hair_thickness_load( target_size, mean_vals, norm_vals,param_path,bin_path);
    }
    env->ReleaseStringUTFChars( model_path_bin, bin_path);
    env->ReleaseStringUTFChars( model_path_param, param_path);
    return result;
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1hairjewelry(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                   jstring dst_img_path, jintArray display_img,
                                                   jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_hairjewelry()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<target_detection_Object>objects;
    g_yoloNcnn_thickness->hair_thickness_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>hairjewelry_areas;
    std::vector<int>hairjewelry_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(hairjewelry_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, hairjewelry_areas.size(), hairjewelry_areas.data());
    jintArray class_num_array = env->NewIntArray(hairjewelry_class.size());
    env->SetIntArrayRegion(class_num_array, 0, hairjewelry_class.size(), hairjewelry_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_hairjewelry"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    hairjewelry_areas.clear();
    hairjewelry_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 9.头发细软

//region 10.毛囊萎缩

extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1follicleatrophy(JNIEnv *env, jclass clazz,
                                                       jstring src_img_path, jstring dst_img_path,
                                                       jintArray display_img,
                                                       jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_follicleatrophy()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<target_detection_Object>objects;
    g_yoloNcnn_thickness->hair_thickness_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>follicleatrophy_areas;
    std::vector<int>follicleatrophy_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(follicleatrophy_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, follicleatrophy_areas.size(), follicleatrophy_areas.data());
    jintArray class_num_array = env->NewIntArray(follicleatrophy_class.size());
    env->SetIntArrayRegion(class_num_array, 0, follicleatrophy_class.size(), follicleatrophy_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_follicleatrophy"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    follicleatrophy_areas.clear();
    follicleatrophy_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 10.毛囊萎缩

//region 11.头皮细纹
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1wrinkle(JNIEnv *env, jclass clazz, jstring src_img_path,
                                               jstring dst_img_path, jintArray display_img,
                                               jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_wrinkle()
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
//    std::vector<target_detection_Object>objects;
//    g_yoloNcnn_thickness->hair_thickness_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>wrinkle_areas;
    std::vector<int>wrinkle_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(wrinkle_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, wrinkle_areas.size(), wrinkle_areas.data());
    jintArray class_num_array = env->NewIntArray(wrinkle_class.size());
    env->SetIntArrayRegion(class_num_array, 0, wrinkle_class.size(), wrinkle_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_wrinkle"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    wrinkle_areas.clear();
    wrinkle_class.clear();
    useImage.release();
//    objects.clear();
//    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 11.头皮细纹

//region 12.头皮水分
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1moisture(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                jstring dst_img_path, jintArray display_img,
                                                jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_moisture()
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
//    std::vector<target_detection_Object>objects;
//    g_yoloNcnn_thickness->hair_thickness_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>moisture_areas;
    std::vector<int>moisture_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(moisture_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, moisture_areas.size(), moisture_areas.data());
    jintArray class_num_array = env->NewIntArray(moisture_class.size());
    env->SetIntArrayRegion(class_num_array, 0, moisture_class.size(), moisture_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_moisture"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    moisture_areas.clear();
    moisture_class.clear();
    useImage.release();
//    objects.clear();
//    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 12.头皮水分

//region 13.红痣
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1nevus(JNIEnv *env, jclass clazz, jstring src_img_path,
                                             jstring dst_img_path, jintArray display_img,
                                             jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_nevus()
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
//    std::vector<target_detection_Object>objects;
//    g_yoloNcnn_thickness->hair_thickness_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>nevus_areas;
    std::vector<int>nevus_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(nevus_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, nevus_areas.size(), nevus_areas.data());
    jintArray class_num_array = env->NewIntArray(nevus_class.size());
    env->SetIntArrayRegion(class_num_array, 0, nevus_class.size(), nevus_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_nevus"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    nevus_areas.clear();
    nevus_class.clear();
    useImage.release();
//    objects.clear();
//    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 13.红痣

//region 14.色斑
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1stain(JNIEnv *env, jclass clazz, jstring src_img_path,
                                             jstring dst_img_path, jintArray display_img,
                                             jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_stain()
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
//    std::vector<target_detection_Object>objects;
//    g_yoloNcnn_thickness->hair_thickness_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>stain_areas;
    std::vector<int>stain_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(stain_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, stain_areas.size(), stain_areas.data());
    jintArray class_num_array = env->NewIntArray(stain_class.size());
    env->SetIntArrayRegion(class_num_array, 0, stain_class.size(), stain_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_stain"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    stain_areas.clear();
    stain_class.clear();
    useImage.release();
//    objects.clear();
//    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 14.色斑

//region 15.肉粒
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1sarcosome(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                 jstring dst_img_path, jintArray display_img,
                                                 jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_sarcosome()
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
//    std::vector<target_detection_Object>objects;
//    g_yoloNcnn_thickness->hair_thickness_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>sarcosome_areas;
    std::vector<int>sarcosome_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(sarcosome_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, sarcosome_areas.size(), sarcosome_areas.data());
    jintArray class_num_array = env->NewIntArray(sarcosome_class.size());
    env->SetIntArrayRegion(class_num_array, 0, sarcosome_class.size(), sarcosome_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_sarcosome"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    sarcosome_areas.clear();
    sarcosome_class.clear();
    useImage.release();
//    objects.clear();
//    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 15.肉粒

//region 16.管状白发
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initWhiteHairModel(JNIEnv *env, jclass clazz, jstring model_path_bin,
                                               jstring model_path_param) {
    // TODO: implement initWhiteHairModel()
    const float mean_vals[] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int target_size = 640;
    config config_;
    int result = config_.model_init_result_succ;
    // reload
    const char *param_path = env->GetStringUTFChars( model_path_param, NULL);
    const char *bin_path = env->GetStringUTFChars( model_path_bin, NULL);
    ncnn::MutexLockGuard g(Lock);
    if (!g_yoloNcnn_whitehair) {
        g_yoloNcnn_whitehair = new YoloNcnn;
        result = g_yoloNcnn_whitehair->white_hair_load( target_size, mean_vals, norm_vals,param_path,bin_path);
    }
    env->ReleaseStringUTFChars( model_path_bin, bin_path);
    env->ReleaseStringUTFChars( model_path_param, param_path);
    return result;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1hairtubular(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                   jstring dst_img_path, jintArray display_img,
                                                   jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_hairtubular()
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<target_detection_Object>objects;
    g_yoloNcnn_whitehair->white_hair_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>tubular_areas;
    std::vector<int>tubular_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(tubular_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, tubular_areas.size(), tubular_areas.data());
    jintArray class_num_array = env->NewIntArray(tubular_class.size());
    env->SetIntArrayRegion(class_num_array, 0, tubular_class.size(), tubular_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_tubular"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    tubular_areas.clear();
    tubular_class.clear();
    useImage.release();
//    objects.clear();
//    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 16.管状白发

//region 17.透状白发
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1hairtransparent(JNIEnv *env, jclass clazz,
                                                       jstring src_img_path, jstring dst_img_path,
                                                       jintArray display_img,
                                                       jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_hairtransparent()
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<target_detection_Object>objects;
    g_yoloNcnn_whitehair->white_hair_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>transparent_areas;
    std::vector<int>transparent_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(transparent_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, transparent_areas.size(), transparent_areas.data());
    jintArray class_num_array = env->NewIntArray(transparent_class.size());
    env->SetIntArrayRegion(class_num_array, 0, transparent_class.size(), transparent_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_transparent"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    transparent_areas.clear();
    transparent_class.clear();
    useImage.release();
//    objects.clear();
//    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 17.透状白发

//region 18.灰状白发
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1hairgrayish(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                   jstring dst_img_path, jintArray display_img,
                                                   jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_hairgrayish()
    // TODO: implement rendering_hairtransparent()
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用 密度
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<target_detection_Object>objects;
    g_yoloNcnn_whitehair->white_hair_detect(useImage,objects,0.1f, 0.1f, 2);

    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>grayish_areas;
    std::vector<int>grayish_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(grayish_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, grayish_areas.size(), grayish_areas.data());
    jintArray class_num_array = env->NewIntArray(grayish_class.size());
    env->SetIntArrayRegion(class_num_array, 0, grayish_class.size(), grayish_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_grayish"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    grayish_areas.clear();
    grayish_class.clear();
    useImage.release();
//    objects.clear();
//    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;
}

//endregion 18.灰状白发

//region 19.毛囊炎症
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initfollicleModel(JNIEnv *env, jclass clazz, jstring model_path_bin,
                                              jstring model_path_param) {
    // TODO: implement initfollicleModel()
    const float mean_vals[] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int target_size = 640;
    config config_;
    int result = config_.model_init_result_succ;
    // reload
    const char *param_path = env->GetStringUTFChars( model_path_param, NULL);
    const char *bin_path = env->GetStringUTFChars( model_path_bin, NULL);
    ncnn::MutexLockGuard g(Lock);
    if (!g_yolo_follicle_inflammation) {
        g_yolo_follicle_inflammation = new Yolo;
        result = g_yolo_follicle_inflammation->follicle_load( target_size, mean_vals, norm_vals,param_path,bin_path);
    }
    env->ReleaseStringUTFChars( model_path_bin, bin_path);
    env->ReleaseStringUTFChars( model_path_param, param_path);
    return result;
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1follicleinflammation(JNIEnv *env, jclass clazz,
                                                            jstring src_img_path,
                                                            jstring dst_img_path,
                                                            jintArray display_img,
                                                            jobjectArray dst_img_name,
                                                            jobject entity) {
    // TODO: implement rendering_follicleinflammation()

    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);

    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    std::vector<Object>objects;
    g_yolo_follicle_inflammation->follicle_detect(useImage,objects,0.1f,0.4f,1);
    g_yolo_follicle_inflammation->follicle_draw(useImage, objects);
    //endregion 模型调用


    //region 函数调用
    CImage_engine* pImageEngine=new CImage_engine();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>follicleinflammation_blockage_areas;
    std::vector<int>follicleinflammation_blockage_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(follicleinflammation_blockage_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, follicleinflammation_blockage_areas.size(), follicleinflammation_blockage_areas.data());
    jintArray class_num_array = env->NewIntArray(follicleinflammation_blockage_class.size());
    env->SetIntArrayRegion(class_num_array, 0, follicleinflammation_blockage_class.size(), follicleinflammation_blockage_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_follicleinflammation"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    follicleinflammation_blockage_areas.clear();
    follicleinflammation_blockage_class.clear();
    useImage.release();
    objects.clear();
    objects.reserve(0);
    //endregion 内存释放区

    return code_enum_.result_succ;

}

//endregion 19.毛囊炎症

//endregion 头皮检测


//region 人脸检测
//region 1.初始化关键点
vector<Point2f> white_keyPoint;
vector<Point2f> negative_keyPoint;
vector<Point2f> positive_keyPoint;
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_initFacemesh(JNIEnv *env, jclass clazz, jfloatArray facemesh_white_x,
                                         jfloatArray facemesh_white_y,
                                         jfloatArray facemesh_negative_x,
                                         jfloatArray facemesh_negative_y,
                                         jfloatArray facemesh_positive_x,
                                         jfloatArray facemesh_positive_y) {
    // TODO: implement initFacemesh()
    //region 初始区
    jfloat *feature_white_datax = (jfloat *) env->GetFloatArrayElements(facemesh_white_x, 0);
    jfloat *feature_white_datay = (jfloat *) env->GetFloatArrayElements(facemesh_white_y, 0);
    jfloat *feature_negative_datax = (jfloat *) env->GetFloatArrayElements(facemesh_negative_x, 0);
    jfloat *feature_negative_datay = (jfloat *) env->GetFloatArrayElements(facemesh_negative_y, 0);
    jfloat *feature_positive_datax = (jfloat *) env->GetFloatArrayElements(facemesh_positive_x, 0);
    jfloat *feature_positive_datay = (jfloat *) env->GetFloatArrayElements(facemesh_positive_y, 0);
    //endregion 初始区

    //region 运算区
    for (int i = 0; i < 478; i++) {
        Point2f point;
        point.x = feature_white_datax[i];
        point.y = feature_white_datay[i];
        white_keyPoint.push_back(point);
    }
    //拿取负偏光特征点
    for (int i = 0; i < 478; i++) {
        Point2f point;
        point.x = feature_negative_datax[i];
        point.y = feature_negative_datay[i];
        negative_keyPoint.push_back(point);
    }
    //拿取正偏光特征点
    for (int i = 0; i < 478; i++) {
        Point2f point;
        point.x = feature_positive_datax[i];
        point.y = feature_positive_datay[i];
        positive_keyPoint.push_back(point);
    }
    //endregion 运算区

    //region 销毁区
    env->ReleaseFloatArrayElements(facemesh_white_x, feature_white_datax, 0);
    env->ReleaseFloatArrayElements(facemesh_white_y, feature_white_datay, 0);
    env->ReleaseFloatArrayElements(facemesh_negative_x, feature_negative_datax, 0);
    env->ReleaseFloatArrayElements(facemesh_negative_y, feature_negative_datay, 0);
    env->ReleaseFloatArrayElements(facemesh_positive_x, feature_positive_datax, 0);
    env->ReleaseFloatArrayElements(facemesh_positive_y, feature_positive_datay, 0);
    //endregion 销毁区

    return code_enum_.result_succ;
}

//endregion 1.初始化关键点

//region 2.人脸油脂

extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1faceoil(JNIEnv *env, jclass clazz, jstring src_img_path,
                                               jstring dst_img_path, jintArray display_img,
                                               jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_faceoil()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);
    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    //endregion 模型调用


    //region 函数调用

    CImage_face* pImageEngine=new CImage_face();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>faceoil_areas;
    std::vector<int>faceoil_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(faceoil_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, faceoil_areas.size(), faceoil_areas.data());
    jintArray class_num_array = env->NewIntArray(faceoil_class.size());
    env->SetIntArrayRegion(class_num_array, 0, faceoil_class.size(), faceoil_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_faceoil"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    faceoil_areas.clear();
    faceoil_class.clear();
    useImage.release();

    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 2.人脸油脂

//region 3.人脸uv斑
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1facestain(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                 jstring dst_img_path, jintArray display_img,
                                                 jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_facestain()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);
    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    //endregion 模型调用


    //region 函数调用

    CImage_face* pImageEngine=new CImage_face();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>facestain_areas;
    std::vector<int>facestain_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(facestain_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, facestain_areas.size(), facestain_areas.data());
    jintArray class_num_array = env->NewIntArray(facestain_class.size());
    env->SetIntArrayRegion(class_num_array, 0, facestain_class.size(), facestain_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_facestain"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    facestain_areas.clear();
    facestain_class.clear();
    useImage.release();

    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 3.人脸uv斑

//region 4.人脸敏感
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1facesensitive(JNIEnv *env, jclass clazz,
                                                     jstring src_img_path, jstring dst_img_path,
                                                     jintArray display_img,
                                                     jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_facesensitive()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);
    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    //endregion 模型调用


    //region 函数调用

    CImage_face* pImageEngine=new CImage_face();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>facesensitive_areas;
    std::vector<int>facesensitive_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(facesensitive_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, facesensitive_areas.size(), facesensitive_areas.data());
    jintArray class_num_array = env->NewIntArray(facesensitive_class.size());
    env->SetIntArrayRegion(class_num_array, 0, facesensitive_class.size(), facesensitive_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_facesensitive"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    facesensitive_areas.clear();
    facesensitive_class.clear();
    useImage.release();

    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 4.人脸敏感

//region 5.人脸棕色
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1facebrown(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                 jstring dst_img_path, jintArray display_img,
                                                 jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_facebrown()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);
    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    //endregion 模型调用


    //region 函数调用

    CImage_face* pImageEngine=new CImage_face();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>facebrown_areas;
    std::vector<int>facebrown_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(facebrown_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, facebrown_areas.size(), facebrown_areas.data());
    jintArray class_num_array = env->NewIntArray(facebrown_class.size());
    env->SetIntArrayRegion(class_num_array, 0, facebrown_class.size(), facebrown_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_facebrown"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    facebrown_areas.clear();
    facebrown_class.clear();
    useImage.release();

    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 5.人脸棕色

//region 6.人脸皱纹
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1facewrinkle(JNIEnv *env, jclass clazz, jstring src_img_path,
                                                   jstring dst_img_path, jintArray display_img,
                                                   jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_facewrinkle()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);
    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    //endregion 模型调用


    //region 函数调用

    CImage_face* pImageEngine=new CImage_face();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>facewrinkle_areas;
    std::vector<int>facewrinkle_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(facewrinkle_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, facewrinkle_areas.size(), facewrinkle_areas.data());
    jintArray class_num_array = env->NewIntArray(facewrinkle_class.size());
    env->SetIntArrayRegion(class_num_array, 0, facewrinkle_class.size(), facewrinkle_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_facebrown"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    facewrinkle_areas.clear();
    facewrinkle_class.clear();
    useImage.release();

    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 6.人脸皱纹

//region 7.人脸卟啉
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_rendering_1faceporphyrin(JNIEnv *env, jclass clazz,
                                                     jstring src_img_path, jstring dst_img_path,
                                                     jintArray display_img,
                                                     jobjectArray dst_img_name, jobject entity) {
    // TODO: implement rendering_faceporphyrin()
    //region 拿取数据
    jclass entityClass = env->FindClass(code_enum_.entityPath.c_str());
    if (!entityClass) {
        return code_enum_.result_fail;
    }
    const char* scrImgPathJava= env->GetStringUTFChars(src_img_path,0);
    const char* dstImgPathJava=env->GetStringUTFChars(dst_img_path,0);
    int displayIndexJava=env->GetArrayLength(display_img);
    jint* displayImgJava=env->GetIntArrayElements(display_img,0);
    jsize dstImgNameIndexJava = env->GetArrayLength(dst_img_name);
    std::vector<std::string>vecDstImgName;
    std::vector<int>vecDisPlayImg;
    std::vector<std::string>vecScrImgName;
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {
        //展示参数
        vecDisPlayImg.push_back(*(displayImgJava+i));
        //输出图片命名
        jstring dstImgNameTemp=(jstring) env->GetObjectArrayElement(dst_img_name, i);;
        const char *dstImageNamecbuf = env->GetStringUTFChars(dstImgNameTemp, 0);
        std::string dstImageNameC = dstImageNamecbuf;
        vecDstImgName.push_back(dstImageNameC);
        env->DeleteLocalRef(dstImgNameTemp);
    }

    env->ReleaseStringUTFChars(src_img_path,scrImgPathJava);
    env->ReleaseStringUTFChars(dst_img_path,dstImgPathJava);
    env->ReleaseIntArrayElements(display_img,displayImgJava,0);
    //endregion 拿取数据

    //region 模型调用
    cv::Mat scrImage=cv::imread(scrImgPathJava);
    cv::Mat useImage=scrImage.clone();
    //endregion 模型调用


    //region 函数调用

    CImage_face* pImageEngine=new CImage_face();
    std::vector<cv::Mat>vecRunScr;
    vecRunScr.push_back(scrImage);
    std::vector<cv::Mat>vecRunResult(dstImgNameIndexJava);
    std::vector<float>faceporphyrin_areas;
    std::vector<int>faceporphyrin_class;
    vecRunResult[0]=scrImage.clone();
    //endregion 函数调用

    //region 判断图片读取和呈现区一一对应
    std::vector<std::string>vecScrDstImgPath;
    //函数调用区
    for (int i = 0; i <dstImgNameIndexJava ; ++i) {

        if(vecDisPlayImg[i]==1)
        {
            cv::imwrite(StringAdd(dstImgPathJava,vecDstImgName[i]),vecRunResult[i]);
        }
        else
        {
            continue;
        }
    }
    jfloatArray areaFeatureArray = env->NewFloatArray(faceporphyrin_areas.size());
    env->SetFloatArrayRegion(areaFeatureArray, 0, faceporphyrin_areas.size(), faceporphyrin_areas.data());
    jintArray class_num_array = env->NewIntArray(faceporphyrin_class.size());
    env->SetIntArrayRegion(class_num_array, 0, faceporphyrin_class.size(), faceporphyrin_class.data());
    jmethodID set_area_arr = env->GetMethodID(entityClass, "setAreaArr", "([F)V");
    env->CallVoidMethod(entity, set_area_arr, areaFeatureArray);
    jmethodID set_categoryArr_arr = env->GetMethodID(entityClass, "setCategoryArr", "([I)V");
    env->CallVoidMethod(entity, set_categoryArr_arr, class_num_array);
    jmethodID set_type = env->GetMethodID(entityClass, "setType", "(Ljava/lang/String;)V");
    env->CallVoidMethod(entity, set_type, env->NewStringUTF("skin_facebrown"));

    //endregion 判断图片读取和呈现区一一对应
    //region 内存释放区
    delete pImageEngine;
    pImageEngine= nullptr;
    vecDstImgName.clear();
    vecDstImgName.reserve(0);
    vecDisPlayImg.clear();
    vecDisPlayImg.reserve(0);
    vecScrImgName.clear();
    vecScrImgName.reserve(0);
    scrImage.release();
    vecRunScr.clear();
    vecRunResult.clear();
    faceporphyrin_areas.clear();
    faceporphyrin_class.clear();
    useImage.release();

    //endregion 内存释放区

    return code_enum_.result_succ;
}
//endregion 7.人脸卟啉

//endregion 人脸检测

//region 销毁模型
extern "C"
JNIEXPORT jint JNICALL
Java_com_zeze_SkinReasoning_destoryModel(JNIEnv *env, jclass clazz) {
    // TODO: implement destoryModel()
    if(g_yolo_follicle)
    {
        if (g_yolo_follicle->yolo.input_indexes().size() != 0)
        {
            delete g_yolo_follicle;
            g_yolo_follicle=nullptr;
        }
    }
    if(g_yolo_segmentation)
    {
        if (g_yolo_segmentation->yolo.input_indexes().size() != 0)
        {
            delete g_yolo_segmentation;
            g_yolo_segmentation=nullptr;
        }
    }

    if(g_yolo_scurf)
    {
        if(g_yolo_scurf->yolo.input_indexes().size()!=0)
        {
            delete g_yolo_scurf;
            g_yolo_scurf= nullptr;
        }
    }

    if(g_yoloncnn_spot)
    {
        if(g_yoloncnn_spot->yolo.input_indexes().size()!=0)
        {
            delete g_yoloncnn_spot;
            g_yoloncnn_spot= nullptr;
        }
    }
    if(g_yoloNcnn_density)
    {
        if(g_yoloNcnn_density->yolo.input_indexes().size()!=0)
        {
            delete g_yoloNcnn_density;
            g_yoloNcnn_density= nullptr;
        }
    }
    if(g_yoloNcnn_thickness)
    {
        if(g_yoloNcnn_thickness->yolo.input_indexes().size()!=0)
        {
            delete g_yoloNcnn_thickness;
            g_yoloNcnn_thickness= nullptr;
        }
    }

    if(g_yoloNcnn_whitehair)
    {
        if(g_yoloNcnn_whitehair->yolo.input_indexes().size()!=0)
        {
            delete g_yoloNcnn_whitehair;
            g_yoloNcnn_whitehair= nullptr;
        }
    }
    if(g_yolo_follicle_inflammation)
    {
        if(g_yolo_follicle_inflammation->yolo.input_indexes().size()!=0)
        {
            delete g_yolo_follicle_inflammation;
            g_yolo_follicle_inflammation= nullptr;
        }
    }


    return code_enum_.result_succ;
}
//endregion 销毁模型


































