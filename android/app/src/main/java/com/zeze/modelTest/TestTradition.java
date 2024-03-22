package com.zeze.modelTest;

import android.graphics.Bitmap;

import com.zeze.SkinReasoning;
import com.zeze.entity.SkinDetectSeqResultEntity;

import java.io.ByteArrayOutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TestTradition {
    public static void detection()
    {

//        long startTime;
//        SkinReasoning.destoryModel();
//        int  result ;
    int[] scalpOil_index={1};
    String[] dstImageName={"dst_oil_1.jpg"};
        SkinDetectSeqResultEntity oilClass=new SkinDetectSeqResultEntity();
        int s=SkinReasoning.rendering_scalpOil("/sdcard/wefen/scrImg/white/1.jpg","/sdcard/wefen/result/",scalpOil_index,dstImageName,oilClass);


    }



    private static String unicodeToString(String str) {

        Pattern pattern = Pattern.compile("(\\\\u(\\p{XDigit}{4}))");
        Matcher matcher = pattern.matcher(str);
        char ch;
        while (matcher.find()) {
            ch = (char) Integer.parseInt(matcher.group(2), 16);
            str = str.replace(matcher.group(1), ch + "");
        }
        return str;
    }
    private static byte[] bitmapToByteArray(Bitmap bitmap){
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        return byteArray;
    }

}
