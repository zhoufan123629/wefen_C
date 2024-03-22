package com.zeze.entity;

public class SkinDetectSeqResultEntity {
    // 坐标
    private float[] positionArr;
    // 面积占比
    private float[] areaArr;
    // 类别占比
    private int[] categoryArr;
    // 图片地址
    private String imgPath;
    private String dstMaskPath;

    // 类型
    private String type;
    private String savePath="/sdcard/hair/result/";
    private String maskPath="/sdcard/hair/result/";
    public float[] getPositionArr() {
        return positionArr;
    }
    public float[] getAreaArr() {
        return areaArr;
    }
    public int[] getCategoryArr() {
        return categoryArr;
    }
    public String getType() {
        return type;
    }
    public void setPositionArr(float[] positionArr) {
        this.positionArr = positionArr;
    }
    public void setAreaArr(float[] areaArr) {
        this.areaArr = areaArr;
    }
    public void setCategoryArr(int[] categoryArr) {
        this.categoryArr = categoryArr;
    }
    public void setImgPath(String imgPath) {
        this.imgPath = imgPath;
    }
    public void setDstMaskImgPath(String dstMaskPath) {
        this.dstMaskPath = dstMaskPath;
    }
    public void setType(String type) {
        this.type = type;
    }
}
