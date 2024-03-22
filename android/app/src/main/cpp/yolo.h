#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>
#include<string>
#include<iostream>
#include <net.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    float theta;
    std::vector<float> mask_feat;
    cv::Mat mask;
};


struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
class Yolo
{
public:
    Yolo();

    int load(const char* modeltype, int target_size,  const float* norm_vals, bool use_gpu = false);


    //白发分割
//    int whitehair_load( int target_size, const float* mean_vals, const float* norm_vals,const char* model_param ,const char* model_bin);
//    int  whitehair_detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f,float class_num=1);
//    int whitehair_Draw(cv::Mat& rgb, const std::vector<Object>& objects);
    //分割头发

    //解决崩溃现象
    //毛囊堵塞
    int follicle_load( int _target_size, const float* _mean_vals, const float* _norm_vals,const char* model_param ,const char* model_bin);
    int follicle_detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f,float class_num=1);
    int follicle_draw(cv::Mat& rgb, const std::vector<Object>& objects);
    //头发分割
    int hair_segmentation_load_new( int _target_size, const float* _mean_vals, const float* _norm_vals,const char* model_param ,const char* model_bin);
    int hair_segmentation_detect_new(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f,float class_num=1);
    int hair_segmentation_Draw_new(cv::Mat& rgb, const std::vector<Object>& objects);

    //角质
    int hair_cutin_load( int target_size, const float* mean_vals, const float* norm_vals,const char* model_param ,const char* model_bin);
    int  hair_cutin_detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f,float class_num=1);
    int hair_cutin_Draw(cv::Mat& rgb, const std::vector<Object>& objects);
    //皱纹分割
    int wrinkle_load( int target_size, const float* mean_vals, const float* norm_vals,const char* model_param ,const char* model_bin);
    int  wrinkle_detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f,float class_num=1);
    int wrinkle_Draw(cv::Mat& rgb, const std::vector<Object>& objects);
    void destory()
    {
        mask_copy.release();
//        blob_pool_allocator.clear();
//        workspace_pool_allocator.clear();
        yolo.clear();
    };



private:
     auto get_chrono_time_one()
    {
        return  std::chrono::high_resolution_clock::now();
    }
public:
    cv::Mat mask_copy;
    ncnn::Net yolo;
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

class Util
{
public:

    static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

    static void qsort_descent_inplace(std::vector<Object>& faceobjects);

    static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool OBB = false);

    static void generate_proposals(std::vector<GridAndStride>& grid_strides, const ncnn::Mat& pred, int num_class, float prob_threshold, std::vector<Object>& objects);

    static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);

    static void decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h, const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad, ncnn::Mat& mask_pred_result,int type=0);
    //解决数量识别多
//    static void decode_mask_(const cv::Mat& mask_feat, const cv::Rect2f& rect, const cv::Mat& mask_proto, const int in_h, const int in_w, const float scale, const int wpad, const int hpad, cv::Mat& mask_pred_result);

    //头发分割
    static void hair_segmentation_decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h, const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad, ncnn::Mat& mask_pred_result);

    static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis, ncnn::Option& opt);

    static void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out, ncnn::Option& opt);

    static void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d, ncnn::Option& opt);

    static void sigmoid(ncnn::Mat& bottom, ncnn::Option& opt);

    static void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob, ncnn::Option& opt);

    static float intersection_area(const Object& a, const Object& b);

    static float intersection_area_obb(const Object& a, const Object& b);

    static float fast_exp(float x);

    static float sigmoid(float x);
};

#endif // YOLO_H
