# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("wefeng")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif ()
# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
include_directories(${CMAKE_SOURCE_DIR}/)
set(ROOT_DIR ${CMAKE_SOURCE_DIR}/../../../)
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
set_target_properties(
        ncnn PROPERTIES
        INTERFACE_COMPILE_OPTIONS "-frtti;-fexceptions"
        #        ncnn.cmake
        #        里面是关的，把它重新打开防止跟opencv2冲突，如果是重新编译ncnn的请自己尝试要开还是关
)

#SHARED动态库 static静态库
add_library( # Sets the name of the library.
        wefeng

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp
        Image_engine.cpp
        yolo.cpp
        yolo_ncnn.cpp
        face.cpp
        code_enum.h)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)
include_directories(
        ${CMAKE_SOURCE_DIR}/ncnn/${ANDROID_ABI}/include/
        ${CMAKE_SOURCE_DIR}/opencv/include/
)
# 设置opencv的动态库
add_library(libopencv_java4 STATIC IMPORTED)
set_target_properties(
        libopencv_java4
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/opencv/${ANDROID_ABI}/libopencv_java4.so
)
# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

#引用lib库
target_link_libraries( # Specifies the target library.
        wefeng
        libopencv_java4
        vulkan
        ncnn
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})