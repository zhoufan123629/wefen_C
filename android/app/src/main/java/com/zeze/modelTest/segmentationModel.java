package com.zeze.modelTest;


import android.content.res.AssetManager;
import android.util.Log;

import com.zeze.SkinReasoning;
import com.zeze.entity.SkinDetectSeqResultEntity;

public class segmentationModel {


    public final static String TAG = "TestModel";

    public static void init() {

        long allLoadModelStartTime = System.currentTimeMillis();
        //region 初始化分割
        //region 1.初始化
//        long startTime = System.currentTimeMillis();
//        int initWrinkleModelResult = SkinReasoning.initWhiteHairModel("sdcard/hair/version_four/hair_white_hair.bin","sdcard/hair/version_four/hair_white_hair.param");
//        Log.e(TAG, "初始化白发:" + (System.currentTimeMillis() - startTime) + "  ---  初始化结果:" + initWrinkleModelResult);
////        endregion
//
////        region 2.初始化毛囊模型
//        startTime = System.currentTimeMillis();
//        int initMandibularModelResult = SkinReasoning.initfollicleModel("sdcard/hair/version_four/hair_follicle.bin","sdcard/hair/version_four/hair_follicle.param");
//        Log.e(TAG, "初始化毛囊:" + (System.currentTimeMillis() - startTime) + "  ---  初始化结果:" + initMandibularModelResult);//1150
//        //endregion
//
//        startTime = System.currentTimeMillis();
//        int initCutinModelResult = SkinReasoning.initHairCutinModel("sdcard/hair/version_four/hair_scurf.bin","sdcard/hair/version_four/hair_scurf.param");
//        Log.e(TAG, "初始化角质:" + (System.currentTimeMillis() - startTime) + "  ---  初始化结果:" + initMandibularModelResult);//1150
        //endregion
//        Log.e(TAG, "初始化分割模型总共耗时:" + ((System.currentTimeMillis() - allLoadModelStartTime)) + "===========================================");
    }


    public static void detection() {

//        SkinReasoning.initClassPath("com/zeze/entity/SkinDetectSeqResultEntity");
        long allLoadModelStartTime = System.currentTimeMillis();
        //region 分割模型推理

        int[] conbinationArr = {1, 0, 1};
//        //白头发
//        long startTime = System.currentTimeMillis();
//        SkinReasoning.skinNcnnWhiteHairArr("sdcard/hair/follicle_photo/version_three/positive/15.jpg","/sdcard/hair/result/", whitehairClass);
//        Log.e(TAG, "白头发推理耗时:" + ((System.currentTimeMillis() - startTime)));//2775
//        float[] whiteHair={0.253f,0.579f};
//        int whiteHairPosition=-2;
//
//        //region 2.毛囊
//         startTime = System.currentTimeMillis();
//        int i = SkinReasoning.skinNcnnfollicleArr("sdcard/hair/follicle_photo/version_three/white/Snapshot016.jpg", "sdcard/hair/follicle_photo/version_three/uv/Snapshot023.jpg", mandibularClass);//5046
//        Log.e(TAG, "毛囊byte推理耗时:" + ((System.currentTimeMillis() - startTime)));
//        Log.e(TAG, "skinNcnnfollicleArr: " + i);
//
//        //endregion
//
//        startTime = System.currentTimeMillis();
//        SkinReasoning.skinNcnnCutinArr("sdcard/hair/follicle_photo/version_three/uv/Snapshot043.jpg", whitehairClass);
//        Log.e(TAG, "角质点击时间:"+(System.currentTimeMillis() - startTime));//154


//        SkinReasoning.destoryModel();
    }


    public static int success_index;


}
