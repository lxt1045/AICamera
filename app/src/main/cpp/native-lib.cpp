#include <jni.h>
#include <string>
#include <algorithm>
#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/core/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"

#include <opencv2/opencv.hpp>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "classes.h"
#define IMG_H 224
#define IMG_W 224
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H *IMG_W *IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "F8DEMO", __VA_ARGS__);

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static char raw_data[MAX_DATA_SIZE];
static float input_data[MAX_DATA_SIZE];
static caffe2::Workspace ws;

AAssetManager *g_mgr =nullptr;

// A function to load the NetDefs from protobufs.
void loadToNetDef(AAssetManager *mgr, caffe2::NetDef *net, const char *filename)
{
    g_mgr = mgr;
    AAsset *asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len))
    {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}

extern "C" void
Java_facebook_f8demo_ClassifyCamera_initCaffe2(
    JNIEnv *env,
    jobject /* this */,
    jobject assetManager)
{
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    alog("Attempting to load protobuf netdefs...");
    /*
    loadToNetDef(mgr, &_initNet, "squeeze_init_net.pb");  //#define IMG_W 227
    loadToNetDef(mgr, &_predictNet, "squeeze_predict_net.pb");
    //*/
    //*
    loadToNetDef(mgr, &_initNet, "bvlc_googlenet/init_net.pb"); //#define IMG_W 224
    loadToNetDef(mgr, &_predictNet, "bvlc_googlenet/predict_net.pb");
    //*/
    /*
    loadToNetDef(mgr, &_initNet, "new_squeeze/exec_net.pb"); //#define IMG_W 256
    loadToNetDef(mgr, &_predictNet, "new_squeeze/predict_net.pb");
    //*/

    alog("done.");
    alog("Instantiating predictor...");
    _predictor = new caffe2::Predictor(_initNet, _predictNet);
    alog("done.")
}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;

extern "C" JNIEXPORT jstring JNICALL
Java_facebook_f8demo_ClassifyCamera_classificationFromCaffe2t(
    JNIEnv *env,
    jobject /* this */,
    jbyteArray png)
{
    if (!_predictor)
    {
        return env->NewStringUTF("Loading...");
    }

    //先获取文件流
    jsize png_len = env->GetArrayLength(png);
    assert(png_len > 0);
    jbyte *png_data = env->GetByteArrayElements(png, 0);

    //全局变量
    std::ostringstream stringStream;  //存返回值的中间值
    cv::Mat bgr_img;
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);   //PNG格式图片的压缩级别
    compression_params.push_back(0);
    std::string   imgFileSavePrePath="/sdcard/1test/";          //SD卡中的临时目录
    try {
        /*
        从assets目录下读文件
        const char *filename =  "imgs/test-resize.png";  //注意：android不支持jpg格式的图片处理。主要是opencv的锅？
        assert(g_mgr != nullptr);
        AAsset *asset = AAssetManager_open(g_mgr, filename, AASSET_MODE_BUFFER);
        assert(asset != nullptr);
        off_t len = AAsset_getLength(asset);
        assert(len != 0);
        std::vector<char> buffer(len,0);
        //assert(AAsset_seek(asset,0,SEEK_SET)>=0);
        assert(AAsset_read(asset,&buffer[0], len)>=0);
        bgr_img = cv::imdecode(cv::Mat(buffer), CV_LOAD_IMAGE_COLOR);   //从文件流中读取图片，转成Mat格式
        AAsset_close(asset);
        //*/
        //*
        //从文件流中读取图片，转成Mat格式
        std::vector<char> buffer(png_data,png_data+png_len);
        bgr_img = cv::imdecode(cv::Mat(buffer), CV_LOAD_IMAGE_COLOR);
        //*/
        // 用opencv的方式直接读入文件
        // bgr_img = cv::imread("/sdcard/1test/imgs/test.png", CV_LOAD_IMAGE_COLOR);

        // 输入图像大小
        const int predHeight = 224; //256;
        const int predWidth = 224;  //256;
        const int crops = 1;		// crops等于1表示batch的数量为1
        const int channels = 3;		// 通道数为3，表示BGR，为1表示灰度图
        const int size = predHeight * predWidth;

        // resize成想要的输入大小
        if(true){
            int height = bgr_img.rows;
            int width = bgr_img.cols;
            const double hscale = ((double)height) / predHeight; // 计算缩放比例
            const double wscale = ((double)width) / predWidth;
            const double scale = hscale < wscale ? hscale : wscale;
            const int newH = predHeight * scale;
            const int newW = predWidth * scale;
            cv::Range Rh( height > newH ? (height - newH)/2 : 0, height > newH ? (height + newH)/2 : height);
            cv::Range Rw( width > newW ? (width - newW)/2 : 0, width > newW ? (width + newW)/2 : width);
            bgr_img = cv::Mat(bgr_img, Rh, Rw);
            //cv::imwrite(imgFileSavePrePath+"imgs/test-crop.png", bgr_img, compression_params);
        }

        //cv::GaussianBlur(bgr_img, bgr_img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);     //高斯滤波降噪
        
        cv::resize(bgr_img, bgr_img, cv::Size{predWidth, predHeight}, 0, 0, cv::INTER_AREA); //缩放到最终大小，否则会导致caffe2异常

        // 这里是将图像复制到连续的存储空间内，用于网络的输入，因为是BGR三通道，所以有三个赋值
        // 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。

        // 初始化网络的输入，因为可能要做batch操作，所以分配一段连续的存储空间
        std::vector<float> inputPlanar(crops * channels * predHeight * predWidth);
        for (auto i = 0; i < predHeight; i++)
        {
            for (auto j = 0; j < predWidth; j++)
            {
                //opencv存储结构是RGB作为一个整体存的，caffe的存储结构是R、G、B分别作为一个通道，三者组成一个张量
                inputPlanar[i * predWidth + j + 0 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 0];
                inputPlanar[i * predWidth + j + 1 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 1];
                inputPlanar[i * predWidth + j + 2 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 2];
            }
        } 

        //一下是caffe2的处理过程
        caffe2::TensorCPU input;
        input.Resize(std::vector<int>({crops, channels, predHeight, predWidth}));
        //input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
        //memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
        input.ShareExternalPointer(inputPlanar.data());

        caffe2::Predictor::TensorVector input_vec{&input};
        caffe2::Predictor::TensorVector output_vec;
        caffe2::Timer t;
        t.Start();
        _predictor->run(input_vec, &output_vec);

        float fps = 1000 / t.MilliSeconds();
        total_fps += fps;
        avg_fps = total_fps / iters_fps;
        total_fps -= avg_fps;

        constexpr int k = 5;
        std::pair<int, float> result[k] ={{0,0.0}}; //最小值放在 result[0] ，其他无序
        // for (auto i = 0; i < k; ++i){
        //     result[i]=std::make_pair(0, 0.0);
        // }
        for (auto output : output_vec)  //注意： 如果一个batch(批次)有多个图片一起处理，则会有多个结果！
        {
            for (auto i = 0; i < output->size(); ++i)  //一张图片的结果
            {
                auto val = output->template data<float>()[i];  //取出结果。每一个标签都会有一个归一化的数值，就是这个结果
                if (val < result[0].second)   ////最小值放在 result[0] ，其他无序
                    continue;
                result[0] = std::make_pair(i, val);
                auto minIndex = 0;
                for (auto j = 1; j < k; ++j)
                {
                    if (result[minIndex].second > result[j].second)
                    {
                        minIndex = j;
                    }
                }
                if (minIndex != 0)
                {
                    result[0] = result[minIndex];
                    result[minIndex] = std::make_pair(i, val);
                }
            }
        }
        //给结果排个序
        std::sort(&result[0], &result[k], [](std::pair<int, float> n1, std::pair<int, float> n2) -> int {
            return n1.second > n2.second;
        });
        std::ostringstream nameStream;  //结果图片存下
        stringStream << avg_fps << " FPS\n";
        for (auto j = 0; j < k; ++j)
        {
            stringStream << j << ": " << imagenet_classes[result[j].first] << " - " << result[j].second *100 << "%\n";
            nameStream << imagenet_classes[result[j].first] << ":" << result[j].second *100<<"-";
        }

        cv::imwrite(imgFileSavePrePath+"imgs/"+nameStream.str()+".png", bgr_img, compression_params);
        return env->NewStringUTF(stringStream.str().c_str());

    }
    catch(std::exception& e)
    {
        stringStream<<"exception:"<<e.what()<<"\n";
        return env->NewStringUTF(stringStream.str().c_str());
    }
    catch(...)
    {
        stringStream<<"exception..."<<std::endl;
        return env->NewStringUTF(stringStream.str().c_str());
    }
}
extern "C" JNIEXPORT jstring JNICALL
Java_facebook_f8demo_ClassifyCamera_classificationFromCaffe2(
   JNIEnv *env,
    jobject /* this */,
    jint h, jint w, jbyteArray Y, jbyteArray U, jbyteArray V,
    jint rowStride, jint pixelStride,
    jboolean infer_HWC)
{
    if (!_predictor)
    {
        return env->NewStringUTF("Loading...");
    }
    jsize Y_len = env->GetArrayLength(Y);
    jbyte *Y_data = env->GetByteArrayElements(Y, 0);
    assert(Y_len <= MAX_DATA_SIZE);
    jsize U_len = env->GetArrayLength(U);
    jbyte *U_data = env->GetByteArrayElements(U, 0);
    assert(U_len <= MAX_DATA_SIZE);
    jsize V_len = env->GetArrayLength(V);
    jbyte *V_data = env->GetByteArrayElements(V, 0);
    assert(V_len <= MAX_DATA_SIZE);

#define min(a, b) ((a) > (b)) ? (b) : (a)
#define max(a, b) ((a) > (b)) ? (a) : (b)

    //取中间区域
    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);

    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H)
    {
        iter_h = h;
    }
    if (w < IMG_W)
    {
        iter_w = w;
    }

    for (auto i = 0; i < iter_h; ++i)
    {
        jbyte *Y_row = &Y_data[(h_offset + i) * w];
        jbyte *U_row = &U_data[(h_offset + i) / 4 * rowStride];
        jbyte *V_row = &V_data[(h_offset + i) / 4 * rowStride];
        for (auto j = 0; j < iter_w; ++j)
        {
            // Tested on Pixel and S7.
            char y = Y_row[w_offset + j];
            char u = U_row[pixelStride * ((w_offset + j) / pixelStride)];
            char v = V_row[pixelStride * ((w_offset + j) / pixelStride)];

            float b_mean = 104.00698793f;
            float g_mean = 116.66876762f;
            float r_mean = 122.67891434f;

            auto b_i = 0 * IMG_H * IMG_W + j * IMG_W + i;
            auto g_i = 1 * IMG_H * IMG_W + j * IMG_W + i;
            auto r_i = 2 * IMG_H * IMG_W + j * IMG_W + i;
//            auto b_i = 0 * IMG_H * IMG_W + i * IMG_W + j;
//            auto g_i = 1 * IMG_H * IMG_W + i * IMG_W + j;
//            auto r_i = 2 * IMG_H * IMG_W + i * IMG_W + j;

            if (infer_HWC)
            {
                b_i = (j * IMG_W + i) * IMG_C;
                g_i = (j * IMG_W + i) * IMG_C + 1;
                r_i = (j * IMG_W + i) * IMG_C + 2;
            }
            /*
                R = Y + 1.402 (V-128)
                G = Y - 0.34414 (U-128) - 0.71414 (V-128)
                B = Y + 1.772 (U-V) 
            */
            input_data[r_i] = -r_mean + (float)((float)min(255., max(0., (float)(y + 1.402 * (v - 128)))));
            input_data[g_i] = -g_mean + (float)((float)min(255., max(0., (float)(y - 0.34414 * (u - 128) - 0.71414 * (v - 128)))));
            input_data[b_i] = -b_mean + (float)((float)min(255., max(0., (float)(y + 1.772 * (u - v)))));
        }
    }

    caffe2::TensorCPU input;
    if (infer_HWC)
    {
        input.Resize(std::vector<int>({IMG_H, IMG_W, IMG_C}));
    }
    else
    {
        input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
    }
    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
    caffe2::Predictor::TensorVector input_vec{&input};
    caffe2::Predictor::TensorVector output_vec;
    caffe2::Timer t;
    t.Start();
    _predictor->run(input_vec, &output_vec);
    float fps = 1000 / t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;

    constexpr int k = 5;
    //float max[k] = {0};                 //最小值放在 max[0] ，其他无序
    std::pair<int, float> result[k] ; //={{0,0.0}}; //最小值放在 result[0] ，其他无序
    for (auto i = 0; i < k; ++i){
        result[i]=std::make_pair(0, 0.0);
    }
    // Find the top-k results manually.
    for (auto output : output_vec)
    {
        for (auto i = 0; i < output->size(); ++i)
        {
            auto val = output->template data<float>()[i];
            if (val < result[0].second)
                continue;
            result[0] = std::make_pair(i, val);
            auto minIndex = 0;
            for (auto j = 1; j < k; ++j)
            {
                if (result[minIndex].second > result[j].second)
                {
                    minIndex = j;
                }
            }
            if (minIndex != 0)
            {
                result[0] = result[minIndex];
                result[minIndex] = std::make_pair(i, val);
            }
        }
    }
    std::sort(&result[0], &result[k], [](std::pair<int, float> n1, std::pair<int, float> n2) -> int {
		return n1.second > n2.second;
	});
    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n";

    for (auto j = 0; j < k; ++j)
    {
        stringStream << j << ": " << imagenet_classes[result[j].first] << " - " << result[j].second * 100 << "%\n";
    }
    return env->NewStringUTF(stringStream.str().c_str());
}
