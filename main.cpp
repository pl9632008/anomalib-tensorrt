#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "cuda_runtime_api.h"
#include "NvInfer.h"

using namespace nvinfer1;

static float in_arr[1*3*256*256];
static float out_arr[1*1*256*256];

class Logger : public ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class Classification{
    public:
        void loadEngine(const std::string & path);
        cv::Mat preprocessImg(cv::Mat& img, const int & input_w, const int & input_h);
        cv::Mat doInference(cv::Mat & org_img);
    private:
        Logger logger_;
        IRuntime * runtime_          =  nullptr;
        ICudaEngine * engine_        =  nullptr;
        IExecutionContext * context_ =  nullptr;

        const int BATCH_SIZE  =  1;
        const int CHANNELS    =  3;
        const int INPUT_H     =  256;
        const int INPUT_W     =  256;
        const int CLASSES     =  256*256;
        const char * input_   =  "input";
        const char * output_  =  "output";
        const float image_threshold = 31.055;
        int  padw = 0 ;
        int  padh = 0 ;
};

void Classification::loadEngine(const std::string & path){
        size_t size{0};
        char * trtModelStream{nullptr};
        std::ifstream file(path, std::ios::binary);

        if(file.good()){
            file.seekg(0,std::ios::end);
            size = file.tellg();
            file.seekg(0,std::ios::beg);
            trtModelStream = new char[size];
            file.read(trtModelStream,size);
            file.close();
        }

        runtime_ = createInferRuntime(logger_);
        engine_ = runtime_->deserializeCudaEngine(trtModelStream,size);
        context_ = engine_->createExecutionContext();

        delete[] trtModelStream;
}

cv::Mat Classification::preprocessImg(cv::Mat& img, const int &input_w, const int & input_h){
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows))); 

    padw = (input_w - w)/2;
    padh = (input_h - h)/2;

    return out;
}


cv::Mat Classification::doInference(cv::Mat & org_img){

        int32_t input_index = engine_->getBindingIndex(input_);
        int32_t output_index = engine_->getBindingIndex(output_);

        void * buffers[2];
        cudaMalloc(&buffers[input_index], BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float));
        cudaMalloc(&buffers[output_index], BATCH_SIZE * CLASSES * sizeof(float));

        cv::Mat pr_img = preprocessImg(org_img, INPUT_W, INPUT_H);

        for(int i = 0; i < INPUT_W * INPUT_H; i++){
            in_arr[i] = (pr_img.at<cv::Vec3b>(i)[2]/255.0 - 0.485)/0.229;
            in_arr[i + INPUT_W * INPUT_H] = (pr_img.at<cv::Vec3b>(i)[1]/255.0 - 0.456)/0.224;
            in_arr[i + 2 * INPUT_W*INPUT_H] = (pr_img.at<cv::Vec3b>(i)[0]/255.0 - 0.406)/0.225;
        }


        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[input_index], in_arr, BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream);
        context_->enqueueV2(buffers, stream, nullptr);
        cudaMemcpyAsync(out_arr, buffers[output_index], BATCH_SIZE * CLASSES * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(buffers[input_index]);
        cudaFree(buffers[output_index]);


        cv::Mat ans(INPUT_H,INPUT_W,CV_32F,out_arr);
        cv::Mat mask = ans>image_threshold;
        cv::Rect mask_rect (int(padw) ,int(padh),  int(INPUT_W-padw*2), int(INPUT_H-padh*2));
        cv::Mat res = mask(mask_rect); 

        cv::Mat org_mask;
        cv::resize(res,org_mask,org_img.size());

        cv::Mat result;
        cv::bitwise_and(org_img,org_img,result,org_mask);

        return result;

}

/*
    "image_threshold": 31.055591583251953,
    "pixel_threshold": 31.055591583251953,
    "min": 3.3780155181884766,
    "max": 46.07868194580078
*/

int main(int argc, char * argv[]){

    Classification cls;

    std::string path = "/wangjiadong/swintest/anomalib.engine";
    cls.loadEngine(path);
    
    cv::Mat img = cv::imread(argv[1]);
    cv::Mat res = cls.doInference(img);
    cv:imwrite("../res.jpg",res);


}
