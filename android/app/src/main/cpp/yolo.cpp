#include "yolo.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "cpu.h"
#include "config.h"
#define TAG "testyoloncnn"
#define LogI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LogD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LogE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define MAX_STRIDE 32
//region 模型推理调用
Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
    blob_pool_allocator.set_size_drop_threshold(1280 * 720 * 3);
    workspace_pool_allocator.set_size_drop_threshold(1280 * 720 * 3);
}

int Yolo::load(const char* modeltype, int _target_size,  const float* _norm_vals, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif
    
    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    yolo.load_param(parampath);
    yolo.load_model(modelpath);

    target_size = _target_size;
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}




//void pretty_print(ncnn::Mat &m, char *objectName)
//{
//    LogE("%s print start", objectName);
//    for (int q=0; q<m.c; q++)
//    {
//        const float *ptr = m.channel(q);
//        for (int y=0; y<m.h; y++)
//        {
//            for (int x = 0; x < m.w; x++) {
//                if(x==m.w/2){
//                    LogE("object[%d][%d][%d] = %f ", y, x, q, ptr[x]);
//                }
//            }
//            ptr += m.w;
//        }
//    }
//    LogE("%s print end", objectName);
//}


//endregion 毛囊炎症

//region 皱纹
int Yolo::wrinkle_load(int _target_size, const float *_mean_vals, const float *_norm_vals,
                       const char *model_param, const char *model_bin) {
    int result=0;
    if(yolo.layers().size()==0)
    {
        yolo.clear();
        blob_pool_allocator.clear();
        workspace_pool_allocator.clear();


        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

        yolo.opt = ncnn::Option();
        yolo.opt.use_bf16_storage = true;
        yolo.opt.use_fp16_packed = true;
        yolo.opt.use_fp16_arithmetic = true;
#if NCNN_VULKAN
        config config_;
        yolo.opt.use_vulkan_compute = config_.use_gpu;
#endif
        yolo.opt.num_threads = ncnn::get_big_cpu_count();
        yolo.opt.blob_allocator = &blob_pool_allocator;
        yolo.opt.workspace_allocator = &workspace_pool_allocator;
        //判断yolo

        result=yolo.load_param(model_param);
        result=yolo.load_model(model_bin);
//        yolo.load_param(mgr, "Hair_density-sim-opt-fp16.param");
//        yolo.load_model(mgr, "Hair_density-sim-opt-fp16.bin");

        int da4=yolo.layers().size();
        target_size =_target_size;
        norm_vals[0] = _norm_vals[0];
        norm_vals[1] = _norm_vals[1];
        norm_vals[2] = _norm_vals[2];
        mean_vals[0] = _mean_vals[0];
        mean_vals[1] = _mean_vals[1];
        mean_vals[2] = _mean_vals[2];
    };


    return result;
}

int Yolo::wrinkle_detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                         float nms_threshold, float class_num) {
    //输入图片宽高
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32 //拿取缩放后的图片
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_REPLICATE, 114.f);//0.f  BORDER_CONSTANT

    in_pad.substract_mean_normalize(0, norm_vals);

    auto start=std::chrono::high_resolution_clock::now();
    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);
    ncnn::Mat out;

    ex.extract("output0", out);
//    ex.extract("/model.24/Constant_10_output_0", out);
//    pretty_print(out,"=========");
    ncnn::Mat mask_proto;
    ex.extract("output1", mask_proto);
    cv::Mat proto = cv::Mat(mask_proto.h, mask_proto.w, CV_32F, (float*)mask_proto.data);
//    reshape1(out, out, 0, out.w, out.h, 0);
//    reshape(mask_proto, mask_proto, 1, 32, -1, 1);
//    reshape1(mask_proto, mask_proto, 0, 32, -1, 0);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    Util::generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    std::vector<Object> proposals;
    std::vector<Object> objects8;
    Util::generate_proposals(grid_strides, out, class_num,prob_threshold, objects8);//报错
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    // sort all proposals by score from highest to lowest
    Util::qsort_descent_inplace(proposals);
    // apply nms with nms_threshold
    std::vector<int> picked;
    Util::nms_sorted_bboxes(proposals, picked, nms_threshold);
    float count = picked.size();
    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        float* mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(), sizeof(float) * proposals[picked[i]].mask_feat.size());
//        free(mask_feat_ptr);
    }
    ncnn::Mat mask_pred_result;
    Util::decode_mask(mask_feat, width, height, mask_proto, in_pad, wpad, hpad, mask_pred_result);
    objects.resize(count);
    mask_copy= cv::Mat(height, width, CV_32FC1);
    mask_copy=cv::Scalar(999);
//    mask_copy=cv::Mat::zeros(height, width, CV_32FC1);
    for (int i = 0; i < count; i++)
    {

        objects[i] = proposals[picked[i]];
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        if(abs(x1-y1)<1)
        {
            x1+=1;
        }
        if(abs(y1-y0)<1)
        {
            y1+=1;
        }
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
//        objects[i].mask = cv::Mat::zeros(height, width, CV_32FC1);
        cv::Mat mask= cv::Mat(mask_pred_result.h, mask_pred_result.w, CV_32FC1, (float*)mask_pred_result.channel(i));
        float mx0 = x0 * scale * 0.25f;
        float mx1 = x1 * scale * 0.25f;
        float my0 = y0 * scale * 0.25f;
        float my1 = y1 * scale * 0.25f;
        if(abs(mx1-mx0)<1)
        {
            mx1+=1;
        }
        if(abs(my1-my0)<1)
        {
            my1+=1;
        }
        if((int)my1 >= mask.rows){
            my1 = mask.rows - 1;
        }
        if((int)mx1 >= mask.cols){
            mx1 = mask.cols - 1;
        }
        cv::Mat roi = mask(cv::Range(my0, my1+1), cv::Range(mx0, mx1+1));
        cv::resize(roi, roi, objects[i].rect.size());
        for(int j=y0, x = 0;j<y1, x < roi.rows;j++, x++)
        {
            for(int k=x0, y = 0;k<x1, y < roi.cols;k++, y++)
            {
                if(roi.at<float>(x,y)>=0.5&& roi.at<float>(x,y)<=1)
                {
                    mask_copy.at<float>(j,k)= objects[i].label;
                }
                else if( mask_copy.at<float>(j,k)==999.0f)
                {
                    mask_copy.at<float>(j,k)=999.0f;
                }
            }
        }
    }
    ex.clear();
    in_pad.release();
    in.release();
    mask_pred_result.release();
    mask_proto.release();
    out.release();
    proposals.clear();
    objects8.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    grid_strides.clear();
    grid_strides.reserve(0);
    mask_feat.release();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return 0;
}

int Yolo::wrinkle_Draw(cv::Mat &rgb, const std::vector<Object> &objects) {
    static const char* class_names[] = {
            "wrinkle"
    };

    static const unsigned char colors[2][3] = {
            {56,  0,   255},
    };

    int color_index = 0;

    cv::Mat dstImage=cv::Mat::zeros(rgb.rows,rgb.cols,rgb.type());



    for(int y = 0; y < rgb.rows; y++){
        unsigned char* image_ptr = dstImage.ptr(y);//rgb
        const float* mask_ptr = mask_copy.ptr<float>(y);
        for(int x = 0; x < rgb.cols; x++){
            for(int k=0;k< 1;k++)
            {
                if(mask_ptr[x] ==k){
                    const unsigned char* color = colors[k];
                    image_ptr[0] = cv::saturate_cast<unsigned char>(image_ptr[0] * 0.5 + color[2] * 0.5);
                    image_ptr[1] = cv::saturate_cast<unsigned char>(image_ptr[1] * 0.5 + color[1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<unsigned char>(image_ptr[2] * 0.5 + color[0] * 0.5);
                    break;
                }
            }

            image_ptr += 3;
        }
    }




    for (int i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        color_index++;


        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
    }
    rgb=dstImage.clone();

    return 0;
}
//endregion  皱纹
//region 毛囊炎症解决崩溃情况

int Yolo::follicle_load(int _target_size, const float *_mean_vals, const float *_norm_vals,
                        const char *model_param, const char *model_bin) {
    int result=0;
    if(yolo.input_indexes().size()==0)
    {
        yolo.clear();
        blob_pool_allocator.clear();
        workspace_pool_allocator.clear();

        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

        yolo.opt = ncnn::Option();
#if NCNN_VULKAN
        config config_;
        yolo.opt.use_vulkan_compute = config_.use_gpu;
#endif
        yolo.opt.num_threads = ncnn::get_big_cpu_count();
        yolo.opt.blob_allocator = &blob_pool_allocator;
        yolo.opt.workspace_allocator = &workspace_pool_allocator;

        result=yolo.load_param(model_param);
        result=yolo.load_model(model_bin);
        int num2=yolo.input_indexes().size();
        target_size = _target_size;
        norm_vals[0] = _norm_vals[0];
        norm_vals[1] = _norm_vals[1];
        norm_vals[2] = _norm_vals[2];
        mean_vals[0] = _mean_vals[0];
        mean_vals[1] = _mean_vals[1];
        mean_vals[2] = _mean_vals[2];

    }

    return result;
}

int Yolo::follicle_detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                          float nms_threshold, float class_num) {
    // letterbox pad to multiple of MAX_STRIDE
    //输入图片宽高
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32 //拿取缩放后的图片
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_REPLICATE, 114.f);//0.f  BORDER_CONSTANT

    in_pad.substract_mean_normalize(0, norm_vals);

    auto start=std::chrono::high_resolution_clock::now();
    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);
    ncnn::Mat out;
    ex.extract("output0", out);
    ncnn::Mat mask_proto;
    ex.extract("output1", mask_proto);
    cv::Mat proto = cv::Mat(mask_proto.h, mask_proto.w, CV_32F, (float*)mask_proto.data);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    Util::generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    std::vector<Object> proposals;
    std::vector<Object> objects8;
    Util::generate_proposals(grid_strides, out, class_num,prob_threshold, objects8);//报错
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());


    // sort all proposals by score from highest to lowest
    Util::qsort_descent_inplace(proposals);
    // apply nms with nms_threshold
    std::vector<int> picked;
    Util::nms_sorted_bboxes(proposals, picked, nms_threshold);
    float count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {

        float* mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(), sizeof(float) * proposals[picked[i]].mask_feat.size());
//        free(mask_feat_ptr);
    }
    ncnn::Mat mask_pred_result;
    Util::decode_mask(mask_feat, width, height, mask_proto, in_pad, wpad, hpad, mask_pred_result);
    objects.resize(count);
    mask_copy= cv::Mat(height, width, CV_32FC1);
    mask_copy=cv::Scalar(999);
//    mask_copy=cv::Mat::zeros(height, width, CV_32FC1);
    for (int i = 0; i < count; i++)
    {

        objects[i] = proposals[picked[i]];
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
        if(abs(x1-y1)<1)
        {
            x1+=1;
        }
        if(abs(y1-y0)<1)
        {
            y1+=1;
        }
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        cv::Mat mask= cv::Mat(mask_pred_result.h, mask_pred_result.w, CV_32FC1, (float*)mask_pred_result.channel(i));
        float mx0 = x0 * scale * 0.25f;
        float mx1 = x1 * scale * 0.25f;
        float my0 = y0 * scale * 0.25f;
        float my1 = y1 * scale * 0.25f;
        if(abs(mx1-mx0)<1)
        {
            mx1+=1;
        }
        if(abs(my1-my0)<1)
        {
            my1+=1;
        }
        if((int)my1 >= mask.rows){
            my1 = mask.rows - 1;
        }
        if((int)mx1 >= mask.cols){
            mx1 = mask.cols - 1;
        }
        cv::Mat roi = mask(cv::Range(my0, my1+1), cv::Range(mx0, mx1+1));
        cv::resize(roi, roi, objects[i].rect.size(),cv::INTER_AREA);
        for(int j=y0, x = 0;j<y1, x < roi.rows;j++, x++)
        {
            for(int k=x0, y = 0;k<x1, y < roi.cols;k++, y++)
            {
                if(roi.at<float>(x,y)>=0.8&& roi.at<float>(x,y)<=1)
                {
                    mask_copy.at<float>(j,k)= objects[i].label;
                }
                else if( mask_copy.at<float>(j,k)==999.0f)
                {
                    mask_copy.at<float>(j,k)=999.0f;
                }
            }
        }
    }
    ex.clear();
    in_pad.release();
    in.release();
    mask_pred_result.release();
    mask_proto.release();
    out.release();
    proposals.clear();
    objects8.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    grid_strides.clear();
    grid_strides.reserve(0);
    mask_feat.release();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return 0;
}

int Yolo::follicle_draw(cv::Mat &rgb, const std::vector<Object> &objects) {
    static const char* class_names[] = {
            "white"
//            ,"red"
    };

    static const unsigned char colors[81][3] = {
            {56,  0,   255},
            {226, 255, 0},

    };

    int color_index = 0;
    cv::Mat dstImage=cv::Mat::zeros(rgb.rows,rgb.cols,rgb.type());



    for(int y = 0; y < rgb.rows; y++){
        unsigned char* image_ptr = dstImage.ptr(y);//rgb
        const float* mask_ptr = mask_copy.ptr<float>(y);
        for(int x = 0; x < rgb.cols; x++){
            for(int k=0;k< 1;k++)
            {
                if(mask_ptr[x]==k){
                    const unsigned char* color = colors[k];
                    image_ptr[0] = cv::saturate_cast<unsigned char>( color[2] );
                    image_ptr[1] = cv::saturate_cast<unsigned char>( color[1] );
                    image_ptr[2] = cv::saturate_cast<unsigned char>( color[0] );
                    break;
                }
            }
            image_ptr += 3;
        }
    }
    for (int i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];
        color_index++;
        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
//        #pragma omp parallel for
//        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
    }

    rgb=dstImage.clone();
    return 0;
}

//endregion 毛囊炎症解决崩溃
//region 头发分割实例化崩溃
int
Yolo::hair_segmentation_load_new(int _target_size, const float *_mean_vals, const float *_norm_vals,
                                 const char *model_param, const char *model_bin) {
    int result=0;
    if(yolo.layers().size()==0)
    {
        yolo.clear();
        blob_pool_allocator.clear();
        workspace_pool_allocator.clear();

        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

        yolo.opt = ncnn::Option();
#if NCNN_VULKAN
        config config_;
        yolo.opt.use_vulkan_compute = config_.use_gpu;
#endif
        yolo.opt.num_threads = ncnn::get_big_cpu_count();
        yolo.opt.blob_allocator = &blob_pool_allocator;
        yolo.opt.workspace_allocator = &workspace_pool_allocator;

        result=yolo.load_param(model_param);
        result=yolo.load_model(model_bin);

        target_size =_target_size;
        norm_vals[0] = _norm_vals[0];
        norm_vals[1] = _norm_vals[1];
        norm_vals[2] = _norm_vals[2];
        mean_vals[0] = _mean_vals[0];
        mean_vals[1] = _mean_vals[1];
        mean_vals[2] = _mean_vals[2];
    }

    return result;
}

int Yolo::hair_segmentation_detect_new(const cv::Mat &rgb, std::vector<Object> &objects,
                                       float prob_threshold, float nms_threshold, float class_num) {
    // letterbox pad to multiple of MAX_STRIDE
    //输入图片宽高
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32 //拿取缩放后的图片
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height,
                                                 w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_REPLICATE, 255.f);//0.f

    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);
    ncnn::Mat out;

    ex.extract("output0", out);

    ncnn::Mat mask_proto;
    ex.extract("output1", mask_proto);
    cv::Mat proto = cv::Mat(mask_proto.h, mask_proto.w, CV_32F, (float *) mask_proto.data);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    Util::generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    std::vector<Object> proposals;
    std::vector<Object> objects8;
    Util::generate_proposals(grid_strides, out, class_num, prob_threshold, objects8);//报错
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());


    // sort all proposals by score from highest to lowest
    Util::qsort_descent_inplace(proposals);
    // apply nms with nms_threshold
    std::vector<int> picked;
    Util::nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    LogE("================分割数据=====count===%d",count);//0

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        float* mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(), sizeof(float) * proposals[picked[i]].mask_feat.size());
//        free(mask_feat_ptr);
    }
    ncnn::Mat mask_pred_result;
    Util::decode_mask(mask_feat, width, height, mask_proto, in_pad, wpad, hpad, mask_pred_result);//修改只适用960 去除裁剪





    objects.resize(count);
    mask_copy= cv::Mat(height, width, CV_32FC1);
    mask_copy=cv::Scalar(999);
//    mask_copy=cv::Mat::zeros(height, width, CV_32FC1);
    LogE("================分割数据=====count===%d",count);//0
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
        if(abs(x1-y1)<1)
        {
            x1+=1;
        }
        if(abs(y1-y0)<1)
        {
            y1+=1;
        }
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
//        objects[i].mask = cv::Mat::zeros(height, width, CV_32FC1);
        cv::Mat mask= cv::Mat(mask_pred_result.h, mask_pred_result.w, CV_32FC1, (float*)mask_pred_result.channel(i));
        float mx0 = x0 * scale * 0.25f;
        float mx1 = x1 * scale * 0.25f;
        float my0 = y0 * scale * 0.25f;
        float my1 = y1 * scale * 0.25f;
        if(abs(mx1-mx0)<1)
        {
            mx1+=1;
        }
        if(abs(my1-my0)<1)
        {
            my1+=1;
        }
        if((int)my1 >= mask.rows){
            my1 = mask.rows - 1;
        }
        if((int)mx1 >= mask.cols){
            mx1 = mask.cols - 1;
        }
        cv::Mat roi = mask(cv::Range(my0, my1 + 1), cv::Range(mx0, mx1 + 1));
//        LogE("============%lf==== roi处理",double(i));//0
//        LogE("==========height:%lf====width:%lf",roi.rows,roi.cols);//0
//        LogE("==========height:%lf====width:%lf",objects[i].rect.size().height,objects[i].rect.size().width);//0
        cv::resize(roi, roi, objects[i].rect.size());
//        LogE("============%lf==== resize处理",double(i));//0
        for(int j=y0, x = 0;j<y1, x < roi.rows;j++, x++)
        {
            for(int k=x0, y = 0;k<x1, y < roi.cols;k++, y++)
            {
                if(roi.at<float>(x,y)>=0.5&& roi.at<float>(x,y)<=1)
                {
                    mask_copy.at<float>(j,k)= objects[i].label;
                }
                else if( mask_copy.at<float>(j,k)==999.0f)
                {
                    mask_copy.at<float>(j,k)=999.0f;
                }
            }
        }
    }

    ex.clear();
    in_pad.release();
    in.release();
    mask_pred_result.release();
    mask_proto.release();
    out.release();
    proposals.clear();
    objects8.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    grid_strides.clear();
    grid_strides.reserve(0);
    mask_feat.release();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return 0;
}

int Yolo::hair_segmentation_Draw_new(cv::Mat &rgb, const std::vector<Object> &objects) {
    static const char* class_names[] = {
            "hair_segmentation"
    };

    static const unsigned char colors[82][3] = {
            {255,0,0},

    };

    int color_index = 0;

    cv::Mat dstImage=cv::Mat::zeros(rgb.rows,rgb.cols,rgb.type());

    for(int y = 0; y < rgb.rows; y++){
        unsigned char* image_ptr = dstImage.ptr(y);//rgb
        const float* mask_ptr = mask_copy.ptr<float>(y);
        for(int x = 0; x < rgb.cols; x++){
            for(int k=0;k< 1;k++)
            {
                if(mask_ptr[x] ==k){
                    const unsigned char* color = colors[k];
                    image_ptr[0] = cv::saturate_cast<unsigned char>(image_ptr[0] * 0.5 + color[2] * 0.5);
                    image_ptr[1] = cv::saturate_cast<unsigned char>(image_ptr[1] * 0.5 + color[1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<unsigned char>(image_ptr[2] * 0.5 + color[0] * 0.5);
                    break;
                }
            }
            image_ptr += 3;
        }
    }



    for (int i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        color_index++;

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
    }
    rgb=dstImage.clone();

    return 0;
}


//endregion 头发分割实例化崩溃
//region 角质
int Yolo::hair_cutin_load(int _target_size, const float *_mean_vals, const float *_norm_vals,
                          const char *model_param, const char *model_bin) {
    int result=0;
    if(yolo.layers().size()==0)
    {
        yolo.clear();
        blob_pool_allocator.clear();
        workspace_pool_allocator.clear();


        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

        yolo.opt = ncnn::Option();
        yolo.opt.use_bf16_storage = true;
        yolo.opt.use_fp16_packed = true;
        yolo.opt.use_fp16_arithmetic = true;
#if NCNN_VULKAN
        config config_;
        yolo.opt.use_vulkan_compute = config_.use_gpu;
#endif
        yolo.opt.num_threads = ncnn::get_big_cpu_count();
        yolo.opt.blob_allocator = &blob_pool_allocator;
        yolo.opt.workspace_allocator = &workspace_pool_allocator;
        //判断yolo

        result=yolo.load_param(model_param);
        result=yolo.load_model(model_bin);
//        yolo.load_param(mgr, "Hair_density-sim-opt-fp16.param");
//        yolo.load_model(mgr, "Hair_density-sim-opt-fp16.bin");

        int da4=yolo.layers().size();
        target_size =_target_size;
        norm_vals[0] = _norm_vals[0];
        norm_vals[1] = _norm_vals[1];
        norm_vals[2] = _norm_vals[2];
        mean_vals[0] = _mean_vals[0];
        mean_vals[1] = _mean_vals[1];
        mean_vals[2] = _mean_vals[2];
    };


    return result;
}

int Yolo::hair_cutin_detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                            float nms_threshold, float class_num) {
    //输入图片宽高
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32 //拿取缩放后的图片
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_REPLICATE, 114.f);//0.f  BORDER_CONSTANT

    in_pad.substract_mean_normalize(0, norm_vals);

    auto start=std::chrono::high_resolution_clock::now();
    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);
    ncnn::Mat out;

    ex.extract("output0", out);
//    ex.extract("/model.24/Constant_10_output_0", out);
//    pretty_print(out,"=========");
    ncnn::Mat mask_proto;
    ex.extract("output1", mask_proto);
    cv::Mat proto = cv::Mat(mask_proto.h, mask_proto.w, CV_32F, (float*)mask_proto.data);
//    reshape1(out, out, 0, out.w, out.h, 0);
//    reshape(mask_proto, mask_proto, 1, 32, -1, 1);
//    reshape1(mask_proto, mask_proto, 0, 32, -1, 0);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    Util::generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    std::vector<Object> proposals;
    std::vector<Object> objects8;
    Util::generate_proposals(grid_strides, out, class_num,prob_threshold, objects8);//报错
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    // sort all proposals by score from highest to lowest
    Util::qsort_descent_inplace(proposals);
    // apply nms with nms_threshold
    std::vector<int> picked;
    Util::nms_sorted_bboxes(proposals, picked, nms_threshold);
    float count = picked.size();
    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        float* mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(), sizeof(float) * proposals[picked[i]].mask_feat.size());
//        free(mask_feat_ptr);
    }
    ncnn::Mat mask_pred_result;
    Util::decode_mask(mask_feat, width, height, mask_proto, in_pad, wpad, hpad, mask_pred_result);
    objects.resize(count);
    mask_copy= cv::Mat(height, width, CV_32FC1);
    mask_copy=cv::Scalar(999);
//    mask_copy=cv::Mat::zeros(height, width, CV_32FC1);
    for (int i = 0; i < count; i++)
    {

        objects[i] = proposals[picked[i]];
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        if(abs(x1-y1)<1)
        {
            x1+=1;
        }
        if(abs(y1-y0)<1)
        {
            y1+=1;
        }
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
//        objects[i].mask = cv::Mat::zeros(height, width, CV_32FC1);
        cv::Mat mask= cv::Mat(mask_pred_result.h, mask_pred_result.w, CV_32FC1, (float*)mask_pred_result.channel(i));
        float mx0 = x0 * scale * 0.25f;
        float mx1 = x1 * scale * 0.25f;
        float my0 = y0 * scale * 0.25f;
        float my1 = y1 * scale * 0.25f;
        if(abs(mx1-mx0)<1)
        {
            mx1+=1;
        }
        if(abs(my1-my0)<1)
        {
            my1+=1;
        }
        if((int)my1 >= mask.rows){
            my1 = mask.rows - 1;
        }
        if((int)mx1 >= mask.cols){
            mx1 = mask.cols - 1;
        }
        cv::Mat roi = mask(cv::Range(my0, my1+1), cv::Range(mx0, mx1+1));
        cv::resize(roi, roi, objects[i].rect.size());
        for(int j=y0, x = 0;j<y1, x < roi.rows;j++, x++)
        {
            for(int k=x0, y = 0;k<x1, y < roi.cols;k++, y++)
            {
                if(roi.at<float>(x,y)>=0.5&& roi.at<float>(x,y)<=1)
                {
                    mask_copy.at<float>(j,k)= objects[i].label;
                }
                else if( mask_copy.at<float>(j,k)==999.0f)
                {
                    mask_copy.at<float>(j,k)=999.0f;
                }
            }
        }
    }
    ex.clear();
    in_pad.release();
    in.release();
    mask_pred_result.release();
    mask_proto.release();
    out.release();
    proposals.clear();
    objects8.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    grid_strides.clear();
    grid_strides.reserve(0);
    mask_feat.release();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return 0;
}

int Yolo::hair_cutin_Draw(cv::Mat &rgb, const std::vector<Object> &objects) {
    static const char* class_names[] = {
            "cutin"
    };

    static const unsigned char colors[2][3] = {
            {56,  0,   255},
    };

    int color_index = 0;

    cv::Mat dstImage=cv::Mat::zeros(rgb.rows,rgb.cols,rgb.type());



    for(int y = 0; y < rgb.rows; y++){
        unsigned char* image_ptr = dstImage.ptr(y);//rgb
        const float* mask_ptr = mask_copy.ptr<float>(y);
        for(int x = 0; x < rgb.cols; x++){
            for(int k=0;k< 1;k++)
            {
                if(mask_ptr[x] ==k){
                    const unsigned char* color = colors[k];
                    image_ptr[0] = cv::saturate_cast<unsigned char>(image_ptr[0] * 0.5 + color[2] * 0.5);
                    image_ptr[1] = cv::saturate_cast<unsigned char>(image_ptr[1] * 0.5 + color[1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<unsigned char>(image_ptr[2] * 0.5 + color[0] * 0.5);
                    break;
                }
            }

            image_ptr += 3;
        }
    }




    for (int i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        color_index++;


        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
    }
    rgb=dstImage.clone();

    return 0;
}


//endregion 角质


//endregion 模型推理


//region 目标检测调用函数

void Util::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void Util::qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void Util::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool OBB)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = OBB ? intersection_area(a, b) : intersection_area_obb(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
//谜底yolov8

void Util::generate_proposals(std::vector<GridAndStride>& grid_strides, const ncnn::Mat& pred, int num_class, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;
            //region yolov8不支持
            obj.mask_feat.resize(32);
            std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
            //endregion
            objects.emplace_back(obj);
        }
    }
}

void Util::generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

void Util::decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h, const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad, ncnn::Mat& mask_pred_result,int type)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;
    ncnn::Mat masks;
    matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks, opt);
    sigmoid(masks, opt);
    reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0, opt);
    slice(masks, mask_pred_result, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2, opt);
    slice(mask_pred_result, mask_pred_result, (hpad / 2) / 4, (in_pad.h - hpad / 2) / 4, 1, opt);
    //如果type==1为语义分割
    if(type==1)
    {
        interp(masks, 4.0, img_w, img_h, mask_pred_result, opt);
    }
}

void Util::slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis, ncnn::Option& opt)
{
    ncnn::Layer* op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);// start
    pd.set(10, ends);// end
    pd.set(11, axes);//axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

void Util::interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out, ncnn::Option& opt)
{
    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);// resize_type
    pd.set(1, scale);// height_scale
    pd.set(2, scale);// width_scale
    pd.set(3, out_h);// height
    pd.set(4, out_w);// width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

void Util::reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d, ncnn::Option& opt)
{
    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w);// start
    pd.set(1, h);// end
    if (d > 0)
        pd.set(11, d);//axes
    pd.set(2, c);//axes
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

void Util::sigmoid(ncnn::Mat& bottom, ncnn::Option& opt)
{
    ncnn::Layer* op = ncnn::create_layer("Sigmoid");

    op->create_pipeline(opt);

    // forward

    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);

    delete op;
}

void Util::matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob, ncnn::Option& opt)
{
    ncnn::Layer* op = ncnn::create_layer("MatMul");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    op->load_param(pd);

    op->create_pipeline(opt);
    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];

    op->destroy_pipeline(opt);

    delete op;
}
float Util::intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

float Util::intersection_area_obb(const Object& a, const Object& b)
{
    float area_r1 = a.rect.area();
    float area_r2 = b.rect.area();
    cv::RotatedRect rect1 = cv::RotatedRect(
            cv::Point((a.rect.x + a.rect.width / 2), (a.rect.y + a.rect.height / 2)),
            cv::Size(a.rect.width, a.rect.height), a.theta);
    cv::RotatedRect rect2 = cv::RotatedRect(
            cv::Point((b.rect.x + b.rect.width / 2), (b.rect.y + b.rect.height / 2)),
            cv::Size(b.rect.width, b.rect.height), b.theta);
    std::vector<cv::Point> intersectingRegion;
    float inter_area;
    cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion);
    if (intersectingRegion.empty())
    {
        inter_area = 0.0f;
    }
    else
    {
        inter_area = cv::contourArea(intersectingRegion);
    }
    return inter_area;
}

float Util::fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

float Util::sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

//void Util::decode_mask_(const ncnn::Mat &mask_feat, const int &img_w, const int &img_h,
//                        const ncnn::Mat &mask_proto, const ncnn::Mat &in_pad, const int &wpad,
//                        const int &hpad, ncnn::Mat &mask_pred_result) {
//
//}


//endregion 目标检测调用函数





