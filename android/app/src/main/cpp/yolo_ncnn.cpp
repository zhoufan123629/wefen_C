#include "yolo_ncnn.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "cpu.h"
#include "config.h"
#define TAG "scalp"
#define LogI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LogD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LogE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define MAX_STRIDE 64


//long long getSystemTime() {
//    struct timeval tv;
//    gettimeofday(&tv, NULL);
//    long long current_timestamp = (long long) tv.tv_sec * 1000 + tv.tv_usec / 1000;
//    return current_timestamp;
//}


void UtilNcnn::qsort_descent_inplace(std::vector<target_detection_Object>& faceobjects, int left, int right)
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

void UtilNcnn::qsort_descent_inplace(std::vector<target_detection_Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void UtilNcnn::nms_sorted_bboxes(const std::vector<target_detection_Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool OBB)
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
        const target_detection_Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const target_detection_Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = OBB ? intersection_area_obb(a, b):intersection_area(a, b) ;
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
void UtilNcnn::hair_thickness_nms_sorted_bboxes(const std::vector<cv::RotatedRect> &rect_object,
                                                std::vector<float> rect_area,
                                                std::vector<int> &picked, float nms_threshold) {
    picked.clear();

    const int n = rect_object.size();

    int num_threshold_new=0.1f;
    for (int i = 0; i < n; i++)
    {
        const cv::RotatedRect a = rect_object[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const cv::RotatedRect& b = rect_object[picked[j]];

            // intersection over union
            //拿取交叉面积
            float inter_area;
            std::vector<cv::Point> intersectingRegion;
            cv::rotatedRectangleIntersection(a, b, intersectingRegion);
            if (intersectingRegion.empty())
            {
                inter_area = 0.0f;
            }
            else
            {
                inter_area = cv::contourArea(intersectingRegion);
            }
            float rect_object_area=rect_area[i];
            float picked_area=rect_area[j];
            // float IoU = inter_area / union_area
            if ((inter_area/rect_object_area)>num_threshold_new||(inter_area/picked_area)>num_threshold_new )
            {
                keep = 0;
            }
            intersectingRegion.clear();
            intersectingRegion.reserve(0);
        }

        if (keep)
            picked.push_back(i);
    }
}
//region 密度yolov8

void UtilNcnn::density_generate_proposals(std::vector<target_detection_GridAndStride>& grid_strides, const ncnn::Mat& pred, int num_class, float prob_threshold, std::vector<target_detection_Object>& objects)
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

            target_detection_Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;
            //region yolov8不支持
            //endregion
            objects.emplace_back(obj);
        }
    }

}

//endregion

void UtilNcnn::generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, int num_class, float prob_threshold, std::vector<target_detection_Object>& Objects)
{
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = feat_blob.channel(q * 8 + 6 + k).row(i)[j];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * 8 + 5).row(i)[j];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(feat_blob.channel(q * 8 + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * 8 + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * 8 + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * 8 + 3).row(i)[j]);
                    float theta = sigmoid(feat_blob.channel(q * 8 + 4).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    target_detection_Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;
                    obj.theta = theta;

                    Objects.push_back(obj);
                }
            }
        }
    }
}
float UtilNcnn::intersection_area(const target_detection_Object& a, const target_detection_Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

float UtilNcnn::intersection_area_obb(const target_detection_Object& a, const target_detection_Object& b)
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

float UtilNcnn::fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

float UtilNcnn::sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}



//endregion 调用函数


YoloNcnn::YoloNcnn()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    blob_pool_allocator.set_size_drop_threshold(1280 * 720 * 3);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_drop_threshold(1280 * 720 * 3);
}

int YoloNcnn::load( int _target_size, const float* _mean_vals, const float* _norm_vals,const char* model_param ,const char* model_bin)
{
    int result=0;
    if (yolo.input_indexes().size() == 0)
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

        yolo.opt.use_fp16_storage = true;
        yolo.opt.use_fp16_arithmetic = true;
        yolo.opt.use_fp16_packed = true;
        yolo.opt.use_sgemm_convolution = true;
        yolo.opt.use_winograd_convolution = true;


        //下颚线
        result=yolo.load_param(model_param);
        result=yolo.load_model(model_bin);
        int num2 = yolo.input_indexes().size();
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


int YoloNcnn::hair_thickness_load( int _target_size, const float *_mean_vals,
                                  const float *_norm_vals,const char* model_param ,const char* model_bin) {
    int result=0;
    int result_bin=0;
    if (yolo.input_indexes().size() == 0)
    {
        yolo.clear();
        blob_pool_allocator.clear();
        workspace_pool_allocator.clear();

        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(ncnn::get_big_cpu_count() * 2);

        yolo.opt = ncnn::Option();
#if NCNN_VULKAN
        config config_;
//        yolo.opt.use_vulkan_compute = config_.use_gpu;
#endif
        yolo.opt.num_threads = ncnn::get_big_cpu_count();
        yolo.opt.blob_allocator = &blob_pool_allocator;
        yolo.opt.workspace_allocator = &workspace_pool_allocator;

        yolo.opt.use_fp16_storage = true;
        yolo.opt.use_fp16_arithmetic = true;
        yolo.opt.use_fp16_packed = true;
//        yolo.opt.use_int8_inference=true;
        yolo.opt.use_sgemm_convolution = true;
        yolo.opt.use_winograd_convolution = true;
        yolo.opt.use_packing_layout = true;
//        yolo.opt.use_vulkan_compute = true;


        //下颚线
        result=yolo.load_param(model_param);
        result_bin=yolo.load_model(model_bin);
        target_size = _target_size;
        norm_vals[0] = _norm_vals[0];
        norm_vals[1] = _norm_vals[1];
        norm_vals[2] = _norm_vals[2];
        mean_vals[0] = _mean_vals[0];
        mean_vals[1] = _mean_vals[1];
        mean_vals[2] = _mean_vals[2];

    }
    if(result==0&&result_bin==0)
    {
        result=0;
    }
    else
    {
        result=-1;
    }
    return result;
}

void pretty_print(ncnn::Mat &m, char *objectName)
{
    //LogE("%s print start", objectName);
    for (int q=0; q<m.c; q++)
    {
        const float *ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x = 0; x < m.w; x++) {
                if(x==m.w/2){
                    //LogE("object[%d][%d][%d] = %f ", y, x, q, ptr[x]);
                }
            }
            ptr += m.w;
        }
    }
    //LogE("%s print end", objectName);
}

//region 目标检测推理

//region 正常目标检测
//region yolov5 目标检测
int YoloNcnn::detect(const cv::Mat& rgb, std::vector<target_detection_Object>& objects, float prob_threshold, float nms_threshold, float class_num)
{
    // letterbox pad to multiple of MAX_STRIDE
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
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
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
       // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = target_size - w;
    int hpad =target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);//0.f
    in_pad.substract_mean_normalize(0, norm_vals);
    // yolov5 model inference
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);

//    pretty_print(in_pad,"===");
    ncnn::Mat out;
   auto start=std::chrono::high_resolution_clock::now();
    ex.extract("out0", out);
    // enumerate all boxes
    auto end=std::chrono::high_resolution_clock::now();
    auto  time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    LogE("================毛囊模型推理=====int_out_3===%lf",(double)(time.count())/1000);//0


    std::vector<target_detection_Object> proposals;
//    proposals.reserve(out.h/);
    for (int i = 0; i < out.h; i++)
    {
        const float* ptr = out.row(i);

        const int num_class = class_num;

        const float cx = ptr[0];
        const float cy = ptr[1];
        const float bw = ptr[2];
        const float bh = ptr[3];
        const float box_score = ptr[4];
        const float* class_scores = ptr + 5;

        // find class index with the biggest class score among all classes
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int j = 0; j < num_class; j++)
        {
            if (class_scores[j] > class_score)
            {
                class_score = class_scores[j];
                class_index = j;
            }
        }

        // combined score = box score * class score
        float confidence = box_score * class_score;

        // filter candidate boxes with combined score >= prob_threshold
        if (confidence < prob_threshold)
            continue;

        // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
        float x0 = cx - bw * 0.5f;
        float y0 = cy - bh * 0.5f;
        float x1 = cx + bw * 0.5f;
        float y1 = cy + bh * 0.5f;

        // collect candidates
        target_detection_Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = class_index;
        obj.prob = confidence;

        proposals.push_back(obj);
    }
    UtilNcnn::qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    UtilNcnn::nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();
    objects.resize(count);
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

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

    }
    in_pad.release();
    in.release();
    ex.clear();
    out.release();
    proposals.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return count;
}
//endregion yolov5

//region yolov8
int YoloNcnn::yolov8_detect(const cv::Mat &rgb, std::vector<target_detection_Object> &objects,
                            float prob_threshold, float nms_threshold, float class_num) {




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
    const int wpad = target_size - w;
    const int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_REPLICATE, 255.f);//0.f  BORDER_CONSTANT

    in_pad.substract_mean_normalize(0, norm_vals);

    auto start=std::chrono::high_resolution_clock::now();
    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);
    ncnn::Mat out;

    ex.extract("output0", out);

    auto end=std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LogE("================毛囊模型推理=====int_out===%lf",(double)(time.count())/1000);//0
//    reshape1(out, out, 0, out.w, out.h, 0);
//    reshape(mask_proto, mask_proto, 1, 32, -1, 1);
//    reshape1(mask_proto, mask_proto, 0, 32, -1, 0);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<target_detection_GridAndStride> grid_strides;
    UtilNcnn::generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    std::vector<target_detection_Object> proposals;
    std::vector<target_detection_Object> objects8;
    UtilNcnn::density_generate_proposals(grid_strides, out, class_num,prob_threshold, objects8);//报错
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    // sort all proposals by score from highest to lowest
    UtilNcnn::qsort_descent_inplace(proposals);
    // apply nms with nms_threshold
    std::vector<int> picked;
    UtilNcnn::nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();

    objects.resize(count);
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

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;


    }
    ex.clear();
    in_pad.release();
    in.release();
    out.release();
    proposals.clear();
    objects8.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    grid_strides.clear();
    grid_strides.reserve(0);
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return 0;


}
//endregion yolov8版本

//endregion

//region 头发粗细目标检测推理

//rect_object存储所有旋转框类，rect_area每个旋转框的面积 ，picked返回出符合条件的索引，nms对应的占比面积,大于过滤
static void hair_thickness_nms_sorted_bboxes(const std::vector<cv::RotatedRect>& rect_object,std::vector<float>rect_area, std::vector<int>& picked, float nms_threshold=0.1)
{
    picked.clear();

    const int n = rect_object.size();


    int nms_threshold_index=0.2f;
    for (int i = 0; i < n; i++)
    {
        const cv::RotatedRect a = rect_object[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const cv::RotatedRect& b = rect_object[picked[j]];

            // intersection over union
            //拿取交叉面积
            float inter_area;
            std::vector<cv::Point> intersectingRegion;
            cv::rotatedRectangleIntersection(a, b, intersectingRegion);
            if (intersectingRegion.empty())
            {
                inter_area = 0.0f;
            }
            else
            {
                inter_area = cv::contourArea(intersectingRegion);
            }
            float rect_object_area=rect_area[i];
            float picked_area=rect_area[j];
            // float IoU = inter_area / union_area
            if ((inter_area/rect_object_area)>nms_threshold_index&&(inter_area/picked_area)>nms_threshold_index )
            {
                keep = 0;
            }
            intersectingRegion.clear();
            intersectingRegion.reserve(0);
        }

        if (keep)
            picked.push_back(i);
    }
}


int
YoloNcnn::hair_thickness_detect(const cv::Mat &rgb, std::vector<target_detection_Object> &objects,
                                float prob_threshold, float nms_threshold, float class_num) {
    // load ncnn model

    // load image, resize and letterbox pad to multiple of max_stride
    const int img_w = rgb.cols;
    const int img_h = rgb.rows;
    const int max_stride = 64;


    // solve resize scale
    int w = img_w;
    int h = img_h;
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

    LogE("================hair===one=====");//0

    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    const int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    const int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
//    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 255.f);
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 128.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    in_pad.substract_mean_normalize(0, norm_vals);
    //LogE("========from_pixels_resize====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);

    // yolov5 model inference
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);
    LogE("================hair===out0=====");//0
    ncnn::Mat out0;
    ncnn::Mat out1;
    ncnn::Mat out2;
    ex.extract("out0", out0);
    ex.extract("out1", out1);
    ex.extract("out2", out2);
    //LogE("========ex====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);
    std::vector<target_detection_Object> proposals;
    LogE("================hair_out========out0===out1===out2");//0
    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat anchors(6);
        anchors[0] = 12.f;
        anchors[1] = 16.f;
        anchors[2] = 19.f;
        anchors[3] = 36.f;
        anchors[4] = 40.f;
        anchors[5] = 28.f;

        std::vector<target_detection_Object> objects8;
        UtilNcnn::generate_proposals(anchors, 8, in_pad, out2,class_num, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        anchors.release();
        objects8.clear();
        objects8.reserve(0);
    }
    int temp=proposals.size();

    // stride 16
    {
        ncnn::Mat anchors(6);
        anchors[0] = 36.f;
        anchors[1] = 75.f;
        anchors[2] = 76.f;
        anchors[3] = 55.f;
        anchors[4] = 72.f;
        anchors[5] = 146.f;

        std::vector<target_detection_Object> objects16;
        UtilNcnn::generate_proposals(anchors, 16, in_pad, out1,class_num, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        anchors.release();
        objects16.clear();
        objects16.reserve(0);
    }
    LogE("=========anchors1======%11d",(proposals.size()-temp));
    temp=proposals.size();
    // stride 32
    {
        ncnn::Mat anchors(6);
        anchors[0] = 142.f;
        anchors[1] = 110.f;
        anchors[2] = 192.f;
        anchors[3] = 243.f;
        anchors[4] = 459.f;
        anchors[5] = 401.f;

        std::vector<target_detection_Object> objects32;
        UtilNcnn::generate_proposals(anchors, 32, in_pad, out0,class_num, prob_threshold, objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        anchors.release();
        objects32.clear();
        objects32.reserve(0);
    }
    LogE("=========anchors2======%11d",(proposals.size()-temp));
    //LogE("========proposals====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);
    // sort all candidates by score from highest to lowest


    UtilNcnn::qsort_descent_inplace(proposals);
    //LogE("========qsort_descent_inplace====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);

//    region 新
//    将所有的框全部附在轮廓上，给每个轮廓标记一下
    std::vector<int> picked;
    UtilNcnn::nms_sorted_bboxes(proposals, picked, nms_threshold, true);
//    endregion 肖
    //region 旧
    //标记之后
//    std::vector<target_detection_Object>copy_proposals= std::vector<target_detection_Object>(proposals);
//    //存储旋转矩形
//    std::vector<cv::RotatedRect >rotated_rect;
//    rotated_rect.resize(copy_proposals.size());
//    std::vector<float>rotated_area;
//    for(int i=0;i<copy_proposals.size();i++)
//    {
//        float theta = (proposals[i].theta -0.5f) * acos(-1);
//        float angle = ((theta * 180) / acos(-1));
//        float x0 = (proposals[i].rect.x - (wpad / 2)) / scale;
//        float y0 = (proposals[i].rect.y - (hpad / 2)) / scale;
//        float x1 = (proposals[i].rect.x + proposals[i].rect.width - (wpad / 2)) / scale;
//        float y1 = (proposals[i].rect.y + proposals[i].rect.height - (hpad / 2)) / scale;
//        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
//        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
//        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
//        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
//        float copy_width=x1-x0;
//        float copy_height=y1-y0;
//        rotated_area.push_back(copy_width*copy_height);
//        cv::Point center_point=cv::Point(x0+copy_width/2,y0+copy_height/2);
//        rotated_rect[i]=cv::RotatedRect(center_point,cv::Size(copy_width,copy_height), angle);
//        //旋转框的面积
//    }
//    //将所有的旋转框进行过滤比较
//    std::vector<int> picked;
//    UtilNcnn::hair_thickness_nms_sorted_bboxes(rotated_rect,rotated_area,picked,nms_threshold);
    //LogE("========hair_thickness_nms_sorted_bboxes====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);
    //endregion
    // collect final result after nms
    const int count = picked.size();
    LogE("=========cout======%11d",count);
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        float theta = (objects[i].theta -0.5f) * acos(-1);

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
        objects[i].theta = theta;
    }
    //LogE("========count====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);
    ex.clear();
    in_pad.release();
    in.release();
    ex.clear();
    out0.release();
    out1.release();
    out2.release();
    proposals.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return 0;
}

//endregion 头发粗细目标检测推理
//region 白发目标检测推理
//region 白灰黑发检测
int YoloNcnn::white_hair_load(int _target_size, const float *_mean_vals, const float *_norm_vals,
                              const char *model_param, const char *model_bin) {
    int result=0;
    if (yolo.input_indexes().size() == 0)
    {
        yolo.clear();
        blob_pool_allocator.clear();
        workspace_pool_allocator.clear();

        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(ncnn::get_big_cpu_count() * 2);

        yolo.opt = ncnn::Option();
#if NCNN_VULKAN
        config config_;
//        yolo.opt.use_vulkan_compute = config_.use_gpu;
#endif
        yolo.opt.num_threads = ncnn::get_big_cpu_count();
        yolo.opt.blob_allocator = &blob_pool_allocator;
        yolo.opt.workspace_allocator = &workspace_pool_allocator;

        yolo.opt.use_fp16_storage = true;
        yolo.opt.use_fp16_arithmetic = true;
        yolo.opt.use_fp16_packed = true;
//        yolo.opt.use_int8_inference=true;
        yolo.opt.use_sgemm_convolution = true;
        yolo.opt.use_winograd_convolution = true;
        yolo.opt.use_packing_layout = true;
//        yolo.opt.use_vulkan_compute = true;


        //下颚线
        result=yolo.load_param(model_param);
        result=yolo.load_model(model_bin);
        int num2 = yolo.input_indexes().size();
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

int YoloNcnn::white_hair_detect(const cv::Mat& rgb, std::vector<target_detection_Object>& objects, float prob_threshold , float nms_threshold ,float class_num) {
    // load image, resize and letterbox pad to multiple of max_stride
    const int img_w = rgb.cols;
    const int img_h = rgb.rows;
    const int max_stride = 64;


    // solve resize scale
    int w = img_w;
    int h = img_h;
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


    clock_t read_time_start=clock();
    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    const int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    const int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    //新
//    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 255.f);
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 255.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    in_pad.substract_mean_normalize(0, norm_vals);
    //LogE("========from_pixels_resize====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);


    read_time_start=clock();
    // yolov5 model inference
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat out0;
    ncnn::Mat out1;
    ncnn::Mat out2;
    ex.extract("out0", out0);
    ex.extract("out1", out1);
    ex.extract("out2", out2);
    //LogE("========ex====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);
    std::vector<target_detection_Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml
    read_time_start=clock();
    // stride 8
    {
        ncnn::Mat anchors(6);
        anchors[0] = 12.f;
        anchors[1] = 16.f;
        anchors[2] = 19.f;
        anchors[3] = 36.f;
        anchors[4] = 40.f;
        anchors[5] = 28.f;

        std::vector<target_detection_Object> objects8;
        UtilNcnn::generate_proposals(anchors, 8, in_pad, out2,class_num, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        anchors.release();
        objects8.clear();
        objects8.reserve(0);
    }
    int temp=proposals.size();

    // stride 16
    {
        ncnn::Mat anchors(6);
        anchors[0] = 36.f;
        anchors[1] = 75.f;
        anchors[2] = 76.f;
        anchors[3] = 55.f;
        anchors[4] = 72.f;
        anchors[5] = 146.f;

        std::vector<target_detection_Object> objects16;
        UtilNcnn::generate_proposals(anchors, 16, in_pad, out1,class_num, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        anchors.release();
        objects16.clear();
        objects16.reserve(0);
    }
    LogE("=========anchors1======%11d",(proposals.size()-temp));
    temp=proposals.size();
    // stride 32
    {
        ncnn::Mat anchors(6);
        anchors[0] = 142.f;
        anchors[1] = 110.f;
        anchors[2] = 192.f;
        anchors[3] = 243.f;
        anchors[4] = 459.f;
        anchors[5] = 401.f;

        std::vector<target_detection_Object> objects32;
        UtilNcnn::generate_proposals(anchors, 32, in_pad, out0,class_num, prob_threshold, objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        anchors.release();
        objects32.clear();
        objects32.reserve(0);
    }
    LogE("=========anchors2======%11d",(proposals.size()-temp));
    //LogE("========proposals====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);
    // sort all candidates by score from highest to lowest

    read_time_start=clock();
    UtilNcnn::qsort_descent_inplace(proposals);
    //LogE("========qsort_descent_inplace====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);

//    region 新
//    将所有的框全部附在轮廓上，给每个轮廓标记一下
    std::vector<int> picked;
    UtilNcnn::nms_sorted_bboxes(proposals, picked, nms_threshold, true);

    read_time_start=clock();
    //endregion
    // collect final result after nms
    const int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        float theta = (objects[i].theta -0.5f) * acos(-1);

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
        objects[i].theta = theta;
    }

    ex.clear();
    in_pad.release();
    in.release();
    ex.clear();
    out0.release();
    out1.release();
    out2.release();
    proposals.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return 0;
}

//endregion 白黑灰发检测
//endregion 白发目标检测推理
//endregion 目标检测推理

//region 敏感分类
int YoloNcnn::sensitive_classification_load(int _target_size, const float *_mean_vals,
                                            const float *_norm_vals, const char *model_param,
                                            const char *model_bin) {
    if (yolo.input_indexes().size() == 0)
    {
        yolo.clear();
        blob_pool_allocator.clear();
        workspace_pool_allocator.clear();

        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(ncnn::get_big_cpu_count() * 2);

        yolo.opt = ncnn::Option();
#if NCNN_VULKAN
        config config_;
//        yolo.opt.use_vulkan_compute = config_.use_gpu;
#endif
        yolo.opt.num_threads = ncnn::get_big_cpu_count();
        yolo.opt.blob_allocator = &blob_pool_allocator;
        yolo.opt.workspace_allocator = &workspace_pool_allocator;

        yolo.opt.use_fp16_storage = true;
        yolo.opt.use_fp16_arithmetic = true;
        yolo.opt.use_fp16_packed = true;
//        yolo.opt.use_int8_inference=true;
        yolo.opt.use_sgemm_convolution = true;
        yolo.opt.use_winograd_convolution = true;
        yolo.opt.use_packing_layout = true;
//        yolo.opt.use_vulkan_compute = true;


        //下颚线
        yolo.load_param(model_param);
        yolo.load_model(model_bin);
        int num2 = yolo.input_indexes().size();
        target_size = _target_size;
        norm_vals[0] = _norm_vals[0];
        norm_vals[1] = _norm_vals[1];
        norm_vals[2] = _norm_vals[2];
        mean_vals[0] = _mean_vals[0];
        mean_vals[1] = _mean_vals[1];
        mean_vals[2] = _mean_vals[2];
    }

    return 0;
}

int YoloNcnn::sensitive_classification_detect(const cv::Mat &rgb,
                                              int  & label,
                                              float prob_threshold, float nms_threshold,
                                              float class_num) {

    // load image, resize and letterbox pad to multiple of max_stride
    const int img_w = rgb.cols;
    const int img_h = rgb.rows;
    const int max_stride = 32;

    // solve resize scale
    int w = img_w;
    int h = img_h;
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


    clock_t read_time_start=clock();
    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    const int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    const int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    in_pad.substract_mean_normalize(0, norm_vals);
    //LogE("========from_pixels_resize====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);


    read_time_start=clock();
    // yolov5 model inference
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);
    std::vector<float> vec_score;
//    std::vector<std::pair<int, float>> tmp;
    vec_score.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        vec_score[j] = out[j];
//        tmp.emplace_back(std::make_pair(j, vec_score[j]));
    }
//    std::sort(tmp.begin(), tmp.end(), [&](std::pair<int, float> a, std::pair<int, float> b) {return a.second > b.second; });
    label = std::max_element(vec_score.begin(), vec_score.end()) - vec_score.begin();

    //LogE("========count====%f",(float)(clock()-read_time_start)/CLOCKS_PER_SEC);
    ex.clear();
    in_pad.release();
    in.release();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return 0;
}



//endregion

//region 1.斑点
int YoloNcnn::spot_load(int _target_size, const float *_mean_vals, const float *_norm_vals,
                        const char *model_param, const char *model_bin) {
    int result=0;
    if (yolo.input_indexes().size() == 0)
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

        yolo.opt.use_fp16_storage = true;
        yolo.opt.use_fp16_arithmetic = true;
        yolo.opt.use_fp16_packed = true;
        yolo.opt.use_sgemm_convolution = true;
        yolo.opt.use_winograd_convolution = true;


        //下颚线
        result=yolo.load_param(model_param);
        result=yolo.load_model(model_bin);
        int num2 = yolo.input_indexes().size();
        target_size = target_size;
        norm_vals[0] = _norm_vals[0];
        norm_vals[1] = _norm_vals[1];
        norm_vals[2] = _norm_vals[2];
        mean_vals[0] = _mean_vals[0];
        mean_vals[1] = _mean_vals[1];
        mean_vals[2] = _mean_vals[2];
    }

    return result;
}

int YoloNcnn::spot_detect(const cv::Mat &rgb, std::vector<target_detection_Object> &objects,
                          float prob_threshold, float nms_threshold, float class_num) {
    // letterbox pad to multiple of MAX_STRIDE
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
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
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = target_size - w;
    int hpad =target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);//0.f
    in_pad.substract_mean_normalize(0, norm_vals);
    // yolov5 model inference
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);

//    pretty_print(in_pad,"===");
    ncnn::Mat out;
    auto start=std::chrono::high_resolution_clock::now();
    ex.extract("out0", out);
    // enumerate all boxes
    auto end=std::chrono::high_resolution_clock::now();
    auto  time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//    LogE("================毛囊模型推理=====int_out_3===%lf",(double)(time.count())/1000);//0


    std::vector<target_detection_Object> proposals;
//    proposals.reserve(out.h/);
    for (int i = 0; i < out.h; i++)
    {
        const float* ptr = out.row(i);

        const int num_class = class_num;

        const float cx = ptr[0];
        const float cy = ptr[1];
        const float bw = ptr[2];
        const float bh = ptr[3];
        const float box_score = ptr[4];
        const float* class_scores = ptr + 5;

        // find class index with the biggest class score among all classes
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int j = 0; j < num_class; j++)
        {
            if (class_scores[j] > class_score)
            {
                class_score = class_scores[j];
                class_index = j;
            }
        }

        // combined score = box score * class score
        float confidence = box_score * class_score;

        // filter candidate boxes with combined score >= prob_threshold
        if (confidence < prob_threshold)
            continue;

        // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
        float x0 = cx - bw * 0.5f;
        float y0 = cy - bh * 0.5f;
        float x1 = cx + bw * 0.5f;
        float y1 = cy + bh * 0.5f;

        // collect candidates
        target_detection_Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = class_index;
        obj.prob = confidence;

        proposals.push_back(obj);
    }
    UtilNcnn::qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    UtilNcnn::nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();
    objects.resize(count);
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

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

    }
    in_pad.release();
    in.release();
    ex.clear();
    out.release();
    proposals.clear();
    proposals.reserve(0);
    picked.clear();
    picked.reserve(0);
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    return count;
}



//endregion 1.斑点


//region yolov8
void UtilNcnn::generate_grids_and_stride(const int target_w, const int target_h,
                                         std::vector<int> &strides,
                                         std::vector<target_detection_GridAndStride> &grid_strides) {
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                target_detection_GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }

}

//endregion

