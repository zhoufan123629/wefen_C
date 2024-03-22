package com.zeze.wefeng;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.zeze.modelTest.TestTradition;
import com.zeze.modelTest.modeTest;
import com.zeze.modelTest.segmentationModel;
import com.zeze.wefeng.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'wefeng' library on application startup.

    //region 初始化权限状态码
    private ActivityMainBinding binding;

    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE};

    //请求状态码
    private static int REQUEST_PERMISSION_CODE = 1;
    //endregion

    private Button mBtnTest;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_PERMISSION_CODE);
        }
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        TextView tv = binding.sampleText;
        mBtnTest = findViewById(R.id.btn_test);
        mBtnTest.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handler.sendEmptyMessage(2);

            }
        });
        // Example of a call to a native method
//        TextView tv = binding.sampleText;
//        tv.setText(stringFromJNI());
    }
    //region 1.调用函数方法
    private long startTime;
    private Handler handler=new Handler(Looper.getMainLooper()){
        @Override
        public void handleMessage(@NonNull Message msg) {
            super.handleMessage(msg);
            //region 1.目标检测模型
            if (msg.what==0){
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        modeTest.initModel();
                        modeTest.detect();
                    }
                }).start();
            }
            //endregion

            //region 2.分割模型
            if (msg.what==1){
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        segmentationModel.init();
                        segmentationModel.detection();
                    }
                }).start();
            }
            //endregion

            //region 3.指标检测
            if (msg.what==2){
                new Thread(new Runnable() {
                    @Override
                    public void run() {
//                        modeTest.detect();
//                        segmentationModel.detection();
                        TestTradition.detection(

                        );
                    }
                }).start();
            }
            //endregion
        }
    };
    //endregion
    /**
     * A native method that is implemented by the 'wefeng' native library,
     * which is packaged with this application.
     */

}