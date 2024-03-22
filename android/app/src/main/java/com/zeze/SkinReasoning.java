package com.zeze;

import com.zeze.entity.SkinDetectSeqResultEntity;

public class SkinReasoning {

    static {
        System.loadLibrary("wefeng");
    }
    public native static int destoryModel();
//    public native static int initClassPath(String javaObjectName);


    //头皮
    //头皮油脂(完整图片路径，输出文件文件夹,人为决定呈现图片[],需要呈现图片名称,类对象去拿取数据)白光
    public static native int rendering_scalpOil(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //油脂堵塞 uv光
    public static native int initOilBlockageModel(String model_path_bin,String model_path_param);
    public static native int rendering_oilBlockage(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //斑点
    public static native int initSpotModel(String model_path_bin,String model_path_param);
    public static native int rendering_spot(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //头皮敏感
    public static native int initSegmentationModel(String model_path_bin,String model_path_param);
    public static native int rendering_scalpsensitivity(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //水油失衡
    public static native int rendering_wateroil(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //角质头屑
    public native static int initScurfModel(String model_path_bin,String model_path_param);
    public static native int rendering_cutinscurf(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //粉状头屑
    public static native int rendering_powderyscurf(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //头发稀疏
    public native static int initDensityModel(String model_path_bin,String model_path_param);
    public static native int rendering_hairsparse(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //头发细软
    public native static int initThicknessModel(String model_path_bin,String model_path_param);
    public static native int rendering_hairjewelry(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //毛囊萎缩
    public static native int rendering_follicleatrophy(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //头皮细纹
    public static native int rendering_wrinkle(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //头皮水分
    public static native int rendering_moisture(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //红痣
    public static native int rendering_nevus(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //色斑
    public static native int rendering_stain(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //肉粒
    public static native int rendering_sarcosome(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //管状白发
    public native static int initWhiteHairModel(String model_path_bin,String model_path_param);
    public static native int rendering_hairtubular(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //透状白发
    public static native int rendering_hairtransparent (String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //灰状白发
    public static native int rendering_hairgrayish (String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //毛囊炎症
    public static native int initfollicleModel(String model_path_bin,String model_path_param);
    public static native int rendering_follicleinflammation (String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //人脸
    public native static int initFacemesh(float[] facemesh_white_x, float[] facemesh_white_y, float[] facemesh_negative_x, float[] facemesh_negative_y, float[] facemesh_positive_x, float[] facemesh_positive_y);
    //人脸油脂
    public static native int rendering_faceoil (String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //人脸uv斑
    public static native int rendering_facestain (String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //人脸敏感图
    public static native int rendering_facesensitive (String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //人脸棕色图
    public static native int rendering_facebrown(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //人脸皱纹图
    public static native int rendering_facewrinkle(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);
    //人脸卟啉
    public static native int rendering_faceporphyrin(String srcImgPath,String dstImgPath,int[] displayImg,String[] dstImgName,SkinDetectSeqResultEntity entity);

}
