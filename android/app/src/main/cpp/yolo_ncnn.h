#ifndef YOLO_NCNN_H
#define YOLO_NCNN_H

#include <opencv2/core/core.hpp>
#include<string>
#include<iostream>
#include <net.h>
#include<time.h>
struct target_detection_Object
{
    cv::Rect_<float> rect;
    int label;
    float theta;
    float prob;
    cv::Mat mask;
    std::vector<float> mask_feat;
};


struct target_detection_GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
class YoloNcnn
{
public:
    YoloNcnn();
    //yolov5密度
    int load(int target_size, const float* mean_vals, const float* norm_vals,const char* model_param ,const char* model_bin);
    int detect(const cv::Mat& rgb, std::vector<target_detection_Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f, float class_num = 1);
    int draw(cv::Mat& rgb, const std::vector<target_detection_Object>& objects);
    //旋转框
    int hair_thickness_load( int target_size, const float* mean_vals, const float* norm_vals,const char* model_param ,const char* model_bin);
    int hair_thickness_detect(const cv::Mat& rgb, std::vector<target_detection_Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f, float class_num = 1);
    int hair_thickness_draw(cv::Mat& rgb, const std::vector<target_detection_Object>& objects);
    //敏感分类
    int sensitive_classification_load( int target_size, const float* mean_vals, const float* norm_vals,const char* model_param ,const char* model_bin);
    int sensitive_classification_detect(const cv::Mat& rgb, int& label, float prob_threshold = 0.4f, float nms_threshold = 0.5f, float class_num = 1);
    //yolov8
    int yolov8_detect(const cv::Mat& rgb, std::vector<target_detection_Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f,float class_num=1);
    //白发检测
    int white_hair_load( int _target_size, const float* _mean_vals, const float* _norm_vals,const char* model_param ,const char* model_bin);
    int white_hair_detect(const cv::Mat& rgb, std::vector<target_detection_Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f, float class_num = 1);
    //斑点检测
    int spot_load(int target_size, const float* mean_vals, const float* norm_vals,const char* model_param ,const char* model_bin);
    int spot_detect(const cv::Mat& rgb, std::vector<target_detection_Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f, float class_num = 1);



    void destory()
    {
        yolo.clear();
    };
    ncnn::Net yolo;
private:

    ncnn::Net yolo_copy;
    float mean_vals[3];
    int target_size;
    float norm_vals[3];
    int image_w;
    int image_h;
    int in_w;
    int in_h;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};
class UtilNcnn
{
public:

    static void qsort_descent_inplace(std::vector<target_detection_Object>& faceobjects, int left, int right);

    static void qsort_descent_inplace(std::vector<target_detection_Object>& faceobjects);

    static void nms_sorted_bboxes(const std::vector<target_detection_Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool OBB = false);

    //头发粗细
    static void hair_thickness_nms_sorted_bboxes(const std::vector<cv::RotatedRect>& rect_object,std::vector<float>rect_area, std::vector<int>& picked, float nms_threshold=0.1);
    static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, int num_class, float prob_threshold, std::vector<target_detection_Object>& Objects);
   //密度yolov8
    static void density_generate_proposals(std::vector<target_detection_GridAndStride>& grid_strides, const ncnn::Mat& pred, int num_class, float prob_threshold, std::vector<target_detection_Object>& objects);
    static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<target_detection_GridAndStride>& grid_strides);
    //endregion

    static float intersection_area(const target_detection_Object& a, const target_detection_Object& b);

    static float intersection_area_obb(const target_detection_Object& a, const target_detection_Object& b);

    static float fast_exp(float x);

    static float sigmoid(float x);
};
#endif // YOLO_NCNN_H