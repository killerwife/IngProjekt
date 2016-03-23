#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <vector> 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  
#include <opencv2/core/cuda.hpp> 
#include <opencv2/objdetect.hpp>
#include <opencv2/cudaobjdetect.hpp>

int main(int argc, char* argv[])
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
    std::string trainface = ".\\haarcascade_fullbody.xml";

    //declaration  
    cv::CascadeClassifier ada_cpu;
    //cv::cuda::CascadeClassifier ada_gpu;

    if (!(ada_cpu.load(trainface)))
    {
        printf(" cpu ada xml load fail! \n");
        return 1;
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
    return 0;
}