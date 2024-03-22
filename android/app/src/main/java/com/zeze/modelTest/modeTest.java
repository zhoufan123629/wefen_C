package com.zeze.modelTest;

import android.content.res.AssetManager;
import android.util.Log;

import com.zeze.SkinReasoning;
import com.zeze.entity.SkinDetectSeqResultEntity;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class modeTest {
    public final static String TAG = "TestModel";

    public static void initModel(){

//        Log.e("===============","============================initmodel");
//        long time_start = System.currentTimeMillis();
//    //region 目标检测初始化模型
//        SkinReasoning.initDensityModel("sdcard/hair/version_four/hair_density.bin","sdcard/hair/version_four/hair_density.param");
//        Log.e(TAG, "初始化密度:" +  (System.currentTimeMillis() - time_start));//602
//
//    //粗细
//        time_start = System.currentTimeMillis();
//        SkinReasoning.initThicknessModel("sdcard/hair/version_four/hair_thickness.bin","sdcard/hair/version_four/hair_thickness.param");
//        Log.e(TAG, "初始化头发粗细:" +  (System.currentTimeMillis() - time_start));//284
//
//
//        //头发分割
//        time_start = System.currentTimeMillis();
//        SkinReasoning.initHairSegmentationModel("sdcard/hair/version_four/hair_segmentation.bin","sdcard/hair/version_four/hair_segmentation.param");
//
//        Log.e(TAG, "初始化头发分割:" +  (System.currentTimeMillis() - time_start));//669
////         头皮屑
//        Log.e("==========================","======目标检测模型加载总耗时:::" + (System.currentTimeMillis() - time_start)+"===========================================");
    }
    public static void detect( )
    {
        int index;
        int[] hairRateCombination={1,0,0};
        SkinDetectSeqResultEntity normal =new SkinDetectSeqResultEntity();

        long  time_start = System.currentTimeMillis();

        //region 密度
//        //region 1.总的密度
//        time_start = System.currentTimeMillis();
//        index=SkinReasoning.densityReasoningDetect("sdcard/hair/follicle_photo/version_three/positive/15.jpg",normal);
//        Log.e(TAG, "密度推理时间:" +  (System.currentTimeMillis() - time_start));
//
//
//        //endregion 密度
//
//        //region 粗细
//        //region 1.总的粗细
//        time_start = System.currentTimeMillis();
//        SkinDetectSeqResultEntity normal_1 =new SkinDetectSeqResultEntity();
//        index=SkinReasoning.thicknessReasoningDetect("sdcard/hair/follicle_photo/version_three/positive/15.jpg",normal_1);
//        Log.e("==========================","头发粗细推理耗时:::" + (System.currentTimeMillis() - time_start));/*1983*///src_partial_1685945934803.jpg
//
//
//        SkinDetectSeqResultEntity sensitive_entity =new SkinDetectSeqResultEntity();
//        SkinDetectSeqResultEntity sensitive_hot =new SkinDetectSeqResultEntity();
//        time_start = System.currentTimeMillis();
//        int[] selset_display={1,1};
//        index=SkinReasoning.sensitivityDetection("sdcard/hair/follicle_photo/version_three/positive/44.jpg",selset_display, sensitive_entity);
//        Log.e(TAG, "敏感绘制时间:" + (System.currentTimeMillis() - time_start));//539
//
//        SkinDetectSeqResultEntity wrinkle_entity =new SkinDetectSeqResultEntity();
//
//        time_start = System.currentTimeMillis();
//        index=SkinReasoning.skinWrinkleDetect("sdcard/hair/follicle_photo/version_three/wrinkle/22.jpg",wrinkle_entity);
//        int[] pos=wrinkle_entity.getCategoryArr();
//        Log.e(TAG,"皱纹头发面积"+pos[0]);
//        float[] area=wrinkle_entity.getAreaArr();
//        float hair_area=0;//头发的面积
//        float all_area=0;//皱纹的面积
//        float all_width=0;//所有识别皱纹宽的和
//        float all_height=0;//所有识别皱纹高的和
//        float max_width=0;//所有识别皱纹中最大宽
//        float max_height=0;//所有识别皱纹中最大高
//        for (int i=0;i<wrinkle_entity.getAreaArr().length;i++)
//        {
//
//            if(wrinkle_entity.getCategoryArr()[i]==0)
//            {
//                hair_area=wrinkle_entity.getAreaArr()[i];//头发面积
//            }
//            else if(wrinkle_entity.getCategoryArr()[i]==1)//高
//            {
//                all_height+=wrinkle_entity.getAreaArr()[i];
//            }
//           else if(wrinkle_entity.getCategoryArr()[i]==2)//宽
//            {
//                max_width+=wrinkle_entity.getAreaArr()[i];
//            }
//            else if(wrinkle_entity.getCategoryArr()[i]==3)
//            {
//                all_area=wrinkle_entity.getAreaArr()[i];//所有皱纹面积
//            }
//            else if(wrinkle_entity.getCategoryArr()[i]==4)
//            {
//                max_height=wrinkle_entity.getAreaArr()[i];//最大高
//            }
//            else {
//
//                max_width=wrinkle_entity.getAreaArr()[i];//最大宽
//            }
////            all_area+=area[i];
//        }
//        Log.e(TAG,"皱纹所有面积"+all_area);
//        Log.e(TAG, "皱纹绘制时间:" + (System.currentTimeMillis() - time_start));//539



//        SkinReasoning.skin_wrinkle_detect()
//
//        index=SkinReasoning.scurfReasoningDetect("/sdcard/hair/scurf/1.jpg",normal);
//        float[] scurfHair={0.08828f ,0.17638f};//0.56953f,0.18055f 红一个毛囊  0.37578f,0.40833f 无
//        int scurfPosition=-2;
//        scurfPosition=SkinReasoning.clickScurfClass(scurfHair,"sdcard/hair/result/dst_mask_hair_scurf.jpg");
//        Log.e(TAG, "头皮屑类比:" + scurfPosition);
//        Log.e("========","头皮屑数量:" + " - " +normal.getPositionArr().length/5);





    }
    public static float[] array;
    public static float[] acne_array;
    public static float[] fat_array;
    public static float[] sclerythrin_array;

    public static int success_index;
    public static int result;

    //存储

    public static void getFloat( SkinDetectSeqResultEntity stainClass,String path)
    {
        float[] pos=stainClass.getPositionArr();
        StringBuffer xResultBuff = new StringBuffer();
//        for(int i=0;i<pos.length;i++)
//        {
//            if(i==0)
//            {
//                xResultBuff.append("{");
//            }
//                xResultBuff.append(pos[i]).append(",");
//            if(i==pos.length-1)
//            {
//                xResultBuff.append("}");
//            }
//
//
//        }
        //十个为一组
        xResultBuff.append( pos.length/10).append("\n");
        for(int i=0;i<pos.length;i++)
        {
            if((i+1)%10==0)
            {
                xResultBuff.append(pos[i]).append("\n");
                continue;
            }

            xResultBuff.append(pos[i]).append(" ");

        }
        saveTextFile(xResultBuff, "/sdcard/hair/result/"+path+".txt");
    }
    public static void saveTextFile(StringBuffer buff, String textFilePath) {
        try {
            File writename = new File(textFilePath); // 相对路径，如果没有则要建立一个新的output。txt文件
            writename.createNewFile(); // 创建新文件
            BufferedWriter out = new BufferedWriter(new FileWriter(writename));
            out.write(buff.toString()); //
            out.flush(); // 把缓存区内容压入文件
            out.close(); // 最后记得关闭文件
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
