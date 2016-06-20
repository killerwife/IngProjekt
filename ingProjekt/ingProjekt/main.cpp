#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector> 
#include <opencv2/core.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  
#include <opencv2/core/cuda.hpp> 
#include <opencv2/objdetect.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include "Parser.h"
#include <stdint.h>

void detection()
{
    //for time measure  
    float TakeTime;
    unsigned long Atime, Btime;

    //window  
    cv::namedWindow("origin");

    //load image  
    cv::Mat img = cv::imread("SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_001812.png");
    cv::Mat grayImg; //adaboost detection is gray input only.  
    cvtColor(img, grayImg, CV_BGR2GRAY);

    //load xml file  
    std::string trainface = "trainedBoost.xml";

    //declaration  
    cv::CascadeClassifier ada_cpu;
    //cv::cuda::CascadeClassifier ada_gpu;

    if (!(ada_cpu.load(trainface)))
    {
        printf(" cpu ada xml load fail! \n");
        return;
    }

    //if (!(ada_gpu.load(trainface)))
    //{
    //    printf(" gpu ada xml load fail! \n");
    //    return;
    //}

    //////////////////////////////////////////////  
    //cpu case face detection code  
    std::vector< cv::Rect > faces;
    Atime = cv::getTickCount();
    ada_cpu.detectMultiScale(grayImg, faces);
    Btime = cv::getTickCount();
    TakeTime = (Btime - Atime) / cv::getTickFrequency();
    printf("detected face(cpu version) = %d / %lf sec take.\n", faces.size(), TakeTime);
    if (faces.size() >= 1)
    {
        for (int ji = 0; ji < faces.size(); ++ji)
        {
            rectangle(img, faces[ji], CV_RGB(0, 0, 255), 4);
        }
    }

    /////////////////////////////////////////////  
    //gpu case face detection code  
    cv::cuda::GpuMat faceBuf_gpu;
    cv::cuda::GpuMat GpuImg;
    GpuImg.upload(grayImg);
    Atime = cv::getTickCount();
    //int detectionNumber = ada_gpu.detectMultiScale(GpuImg, faceBuf_gpu);
    Btime = cv::getTickCount();
    TakeTime = (Btime - Atime) / cv::getTickFrequency();
    //printf("detected face(gpu version) =%d / %lf sec take.\n", detectionNumber, TakeTime);
    cv::Mat faces_downloaded;
    //if (detectionNumber >= 1)
    //{
    //    faceBuf_gpu.colRange(0, detectionNumber).download(faces_downloaded);
    //    cv::Rect* faces = faces_downloaded.ptr< cv::Rect>();


    //    for (int ji = 0; ji < detectionNumber; ++ji)
    //    {
    //        rectangle(img, cv::Point(faces[ji].x, faces[ji].y), cv::Point(faces[ji].x + faces[ji].width, faces[ji].y + faces[ji].height), CV_RGB(255, 0, 0), 2);
    //    }
    //}


    /////////////////////////////////////////////////  
    //result display  
    imshow("origin", img);
    cv::waitKey(0);
}

static cv::Ptr<cv::ml::TrainData>
prepare_train_data(const cv::Mat& data, const cv::Mat& responses, int ntrain_samples)
{
    cv::Mat sample_idx = cv::Mat::zeros(1, data.rows, CV_8U);
    cv::Mat train_samples = sample_idx.colRange(0, ntrain_samples);
    train_samples.setTo(cv::Scalar::all(1));

    int nvars = data.cols;
    cv::Mat var_type(nvars + 1, 1, CV_8U);
    var_type.setTo(cv::Scalar::all(cv::ml::VAR_ORDERED));
    var_type.at<uchar>(nvars) = cv::ml::VAR_CATEGORICAL;

    return cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, responses,
        cv::noArray(), sample_idx, cv::noArray(), var_type);
}

void trainint()
{
    Parser parser;
    cv::Mat* data = nullptr;
    cv::Mat* responses = nullptr;
    std::string tempPos = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Training\\hrac\\RealData\\";
    std::string tempNeg = "C:\\GitHubCode\\anotovanie\\TrainingData\\";
    parser.toMat(&data, &responses, tempPos, tempNeg, 500, 4000);
    cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();
    //cv::Ptr<cv::ml::TrainData> trainData = prepare_train_data(*data,*responses,40);
    //cv::FileStorage fs1("data.yml", cv::FileStorage::WRITE);
    //fs1 << "yourMat" << *data;
    //cv::FileStorage fs2("responses.yml", cv::FileStorage::WRITE);
    //fs2 << "yourMat" << *responses;
    boost->setBoostType(cv::ml::Boost::REAL);
    boost->setWeakCount(100);
    boost->setWeightTrimRate(0.95);
    boost->setMaxDepth(1);
    boost->setUseSurrogates(false);
    boost->setCVFolds(0);
    boost->train(cv::Mat(*data), cv::ml::ROW_SAMPLE, cv::Mat(*responses)); // 'prepare_train_data' returns an instance of ml::TrainData class
    boost->save("trainedBoost.xml");
    delete data;
    delete responses;
}

void detect()
{
    Parser parser;
    cv::Mat* data = nullptr;
    cv::Mat* responses = nullptr;
    std::string tempPos = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Testing\\hrac\\RealData\\";
    std::string tempNeg = "C:\\GitHubCode\\anotovanie\\TrainingData\\";
    parser.toMat(&data, &responses, tempPos, tempNeg, 5000, 10000);
    cv::String filename = "trainedBoost.xml";
    cv::Ptr<cv::ml::Boost> boost = cv::Algorithm::load<cv::ml::Boost>(filename);
    std::vector< cv::Rect > faces;
    cv::Mat result;
    boost->predict(*data, result);
    float *dataz = (float*)result.data;
    long *resultz = (long*)responses->data;
    //for (int i = 0; i < data->rows; i++)
    //{
    //	printf("Poradove cislo:%d Vysledok:%f Spravny Vysledok: %u\n", i, dataz[i], resultz[i]);
    //}
    int spravne = 0, nespravne = 0;
    for (int i = 0; i < data->rows; i++)
    {
        if (dataz[i] == resultz[i]) spravne++;
        else nespravne++;
    }
    printf("Spravnych vysledkov:%d Nespravnych:%d\n", spravne, nespravne);
}

void detectMultiScale()
{
    cv::String filename = "trainedBoost.xml";
    cv::Ptr<cv::ml::Boost> boost = cv::Algorithm::load<cv::ml::Boost>(filename);
    cv::Mat img = cv::imread("C:\\GitHubCode\\anotovanie\\SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_001768.png");
    cv::Mat result(img);
	int shift = 4;
    for (int scale = 4; scale <= 512; scale *= 1.25,shift*=1.25)
    {
		printf("%d\n",scale);
        for (int i = 0; i + scale * 3 < img.cols; i += shift)
        {
            for (int k = 0; k + scale * 5 < img.rows; k += shift)
            {
                cv::Rect rectangleZone(i, k, scale*3, scale * 5);
                cv::Mat imagePart = cv::Mat(img, rectangleZone);
                cv::resize(imagePart, imagePart, cv::Size(96, 160));
                imagePart.convertTo(imagePart, CV_32F);
                imagePart = imagePart.reshape(1, 1);
                cv::Mat response;
                boost->predict(imagePart, response);
                long* responses = (long*)response.data;
                if (responses[0] != 0)
                    rectangle(result, rectangleZone, (0, 0, 255), 2);
            }
        }
    }
    imshow("origin", result);
    cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    //trainint();
    //detect();
    detectMultiScale();
    //trainint();
    //Parser parser;
    //parser.parseNegatives();
    //parser.parsePositives();
    return 0;
}