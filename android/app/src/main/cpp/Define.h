
#pragma once
#include <iostream>
#include<vector>
#include<string>
//#include "jsoncpp/json/json.h"
#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/photo.hpp>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <stdio.h>
#include <algorithm>
#include<memory>
#include <time.h>

using namespace std;
using namespace cv;
typedef unsigned char byte;
//#include<json/json.h>
//using namespace Json;

//extern "C" {
//#include <stdio.h>
//#include "cJSON.h"
//}
//转换平均值
enum
{
	//等于，小于，大于
	E_WORTH_EQUAL,
	E_WORTH_LESSER,
	E_WORTH_MORE
};
enum


{
	E_IMAGE_WHITE,
	E_IMAGE_UV,
	E_IMAGE_POSITIVE_POLARIZED,
	E_IMAGE_BLUE,
	E_IMAGE_NEGATIVE_POLARIZED
};


