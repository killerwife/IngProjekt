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
#include <thread>

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

void fillData(cv::Mat** data, cv::Mat** responses, bool backfitting, int countPos, int countNeg, int countBackfit = 0)
{
    Parser parser;
    std::string tempPos = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Testing\\hrac\\RealData\\";
    std::string tempNeg = "C:\\GitHubCode\\anotovanie\\TrainingData\\";
    std::string tempNegBackfit = "C:\\GitHubCode\\backfitting\\";
    *data = new cv::Mat(0, 0, CV_32S);
    *responses = new cv::Mat(0, 0, CV_32S);
    int arrayPos[1] = { 0 };
    cv::Mat pos(1, 1, CV_32S, arrayPos);
    parser.toMat(data, responses, tempPos, countPos, pos);
    int arrayNeg[1] = { 1 };
    cv::Mat neg(1, 1, CV_32S, arrayNeg);
    parser.toMat(data, responses, tempNeg, countNeg, neg);
    if (backfitting)
        parser.toMat(data, responses, tempNegBackfit, countBackfit, neg);
}

void trainint(bool backfitting, std::string xml)
{
    cv::Mat* data = nullptr;
    cv::Mat* responses = nullptr;
    fillData(&data, &responses, backfitting, 5000, 20000, 8000);
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
    boost->save(xml);
    delete data;
    delete responses;
}

void detect(bool backfitting)
{
    Parser parser;
    cv::Mat* data = nullptr;
    cv::Mat* responses = nullptr;
    fillData(&data, &responses, backfitting, 40000, 40000, 40000);
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

struct RectComparator
{
    bool operator()(cv::Rect a, cv::Rect b)
    {
        return (a.y + a.height) < (b.y + b.height);
    }
};

std::vector<cv::Rect> nonMaxSuppression(std::vector<cv::Rect> boundingBoxes, float overlapThreshold)
{
    std::vector<cv::Rect> result;
    std::sort(boundingBoxes.begin(), boundingBoxes.end(), RectComparator());
    /*for (cv::Rect rect : boundingBoxes)
        std::cout << rect << "\t";*/
    std::vector<float> areas;
    std::vector<int> indexes;
    std::vector<int> x;
    std::vector<int> y;
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<int> weights;
    int i = 0;
    for (cv::Rect box : boundingBoxes)
    {
        indexes.push_back(i++);
        x.push_back(box.x);
        y.push_back(box.y);
        widths.push_back(box.width);
        heights.push_back(box.height);
        areas.push_back((box.height + 1)*(box.width + 1));
        weights.push_back(1);
    }
    //while (indexes.size() > 0)
    //{
    //	int last = indexes.size() - 1;
    //	auto i = indexes[last];
    //	result.push_back(boundingBoxes[i]);
    //	std::vector<int> supress;
    //	supress.push_back(indexes[last]);
    //	for (int j : indexes)
    //	{
    //		int xx1 = x[i] > x[j] ? x[i] : x[j];
    //		int yy1 = y[i] > y[j] ? y[i] : y[j];
    //		int xx2 = x[i] + width[i] < x[j] + width[j] ? x[i] + width[i] : x[j] + width[j];
    //		int yy2 = y[i] + height[i] < x[j] + height[j] ? x[i] + height[i] : x[j] + height[j];
    //		float w = xx2 - xx1 + 1 > 0 ? xx2 - xx1 + 1 : 0;
    //		float h = yy2 - yy1 + 1 > 0 ? yy2 - yy1 + 1 : 0;
    //		float overlap = w*h / areas[j];
    //		if (overlap > overlapThreshold)
    //			supress.push_back(j);
    //	}
    //	for (int removalIdx : supress)
    //		indexes.erase(std::remove(indexes.begin(), indexes.end(), removalIdx), indexes.end());
    //}
    for (int i = 0; i < boundingBoxes.size();)
    {
        float maxOverlap = 0;
        float minSizeDiff = 2;
        int candidate = -1;
        for (int j = 0; j < boundingBoxes.size(); j++)
        {
            if (i == j || (x[i]>x[j] + widths[j] || x[i] + widths[i]<x[j]) || (y[i]>y[j] + heights[j] || y[i] + heights[i] < y[j]))
                continue;

            int xx1 = x[i] > x[j] ? x[i] : x[j];
            int yy1 = y[i] > y[j] ? y[i] : y[j];
            int xx2 = x[i] + widths[i] < x[j] + widths[j] ? x[i] + widths[i] : x[j] + widths[j];
            int yy2 = y[i] + heights[i] < x[j] + heights[j] ? x[i] + heights[i] : x[j] + heights[j];
            float w = xx2 - xx1 + 1 > 0 ? xx2 - xx1 + 1 : 0;
            float h = yy2 - yy1 + 1 > 0 ? yy2 - yy1 + 1 : 0;
            float overlap = w*h / areas[j];
            float areaDiff = areas[i] / areas[j];
            if (overlap > overlapThreshold && (areaDiff > 0.5 && areaDiff < 2))
            {
                if (maxOverlap < overlap || (maxOverlap == overlap && abs(minSizeDiff - 1) > abs(areaDiff - 1)))
                {
                    candidate = j;
                    maxOverlap = overlap;
                    minSizeDiff = areaDiff;
                }
            }
            //printf("%d\t", j);
        }
        //printf("\n%d\n", i);
        if (candidate != -1)
        {
            int height;
            int width;
            if (boundingBoxes[i].area() < boundingBoxes[candidate].area())
            {
                height = boundingBoxes[candidate].height;
                width = boundingBoxes[candidate].width;
            }
            else
            {
                height = boundingBoxes[i].height;
                width = boundingBoxes[i].width;
            }
            int finalWeight = weights[candidate] + weights[i];
            int middleFirstX = boundingBoxes[i].x + boundingBoxes[i].width / 2;
            int middleSecondX = (boundingBoxes[candidate].x + boundingBoxes[candidate].width / 2);
            int middleFirstY = boundingBoxes[i].y + boundingBoxes[i].height / 2;
            int middleSecondY = (boundingBoxes[candidate].y + boundingBoxes[candidate].height / 2);
            int middleX = (int)((double)abs(middleFirstX - middleSecondX) / finalWeight*weights[candidate]);
            int middleY = (int)((double)abs(middleFirstY - middleSecondY) / finalWeight*weights[candidate]);
            int coordX = middleX - width / 2 + boundingBoxes[i].x>boundingBoxes[candidate].x ? boundingBoxes[candidate].x : boundingBoxes[i].x;
            int coordY = middleY - height / 2 + boundingBoxes[i].y>boundingBoxes[candidate].y ? boundingBoxes[candidate].y : boundingBoxes[i].y;
            boundingBoxes.push_back(cv::Rect(coordX, coordY, width, height));
            areas.push_back((height + 1)*(width + 1));
            x.push_back(coordX);
            y.push_back(coordY);
            widths.push_back(width);
            heights.push_back(height);
            weights.push_back(finalWeight);
            boundingBoxes.erase(boundingBoxes.begin() + i);
            areas.erase(areas.begin() + i);
            x.erase(x.begin() + i);
            y.erase(y.begin() + i);
            widths.erase(widths.begin() + i);
            heights.erase(heights.begin() + i);
            weights.erase(weights.begin() + i);
            if (i < candidate)
                candidate--;
            boundingBoxes.erase(boundingBoxes.begin() + candidate);
            areas.erase(areas.begin() + candidate);
            x.erase(x.begin() + candidate);
            y.erase(y.begin() + candidate);
            widths.erase(widths.begin() + candidate);
            heights.erase(heights.begin() + candidate);
            weights.erase(weights.begin() + candidate);
            //printf("Joined %d and %d\n", i, candidate);
            i = 0;
        }
        else
            i++;
    }
    return boundingBoxes;
}

void detectMultiScale(bool exportShit, std::string xml, std::string filename, std::string imageName)
{
    cv::Ptr<cv::ml::Boost> boost = cv::Algorithm::load<cv::ml::Boost>(xml);
    cv::Mat img = cv::imread("C:\\GitHubCode\\anotovanie\\" + imageName);
    cv::Mat result = img.clone();
    std::vector<cv::Rect> boundingBoxes;
    FILE * file = fopen("rects.txt", "w");
    int shift = 4;
    int m = 8009;
    for (int scale = 8; scale <= 512; scale *= 1.25, shift *= 1.25)
    {
        printf("%d\n", scale);
        for (int i = 0; i + scale * 3 < img.cols; i += shift)
        {
            for (int k = 0; k + scale * 5 < img.rows; k += shift)
            {
                cv::Rect rectangleZone(i, k, scale * 3, scale * 5);
                cv::Mat imagePart = cv::Mat(img, rectangleZone);
                cv::resize(imagePart, imagePart, cv::Size(96, 160));
                imagePart.convertTo(imagePart, CV_32F);
                cv::Mat imagePartInput = imagePart.reshape(1, 1);
                cv::Mat response;
                boost->predict(imagePartInput, response);
                long* responses = (long*)response.data;
                if (responses[0] == 0)
                {
                    //rectangle(result, rectangleZone, (0, 0, 255), 2);
                    boundingBoxes.push_back(rectangleZone);
                    fprintf(file, "%d %d %d %d\n", rectangleZone.x, rectangleZone.y, rectangleZone.width, rectangleZone.height);
                    if (exportShit)
                    {
                        cv::imwrite("C:\\GitHubCode\\backfitting\\pic" + std::to_string(m) + ".png", imagePart);
                        m++;
                    }

                }
            }
        }
    }
    auto resultBoundingBoxes = nonMaxSuppression(boundingBoxes, 0.3);
    for (cv::Rect& box : resultBoundingBoxes)
    {
        rectangle(result, box, (0, 0, 255), 2);
    }
    cv::imwrite(filename, result);
    cv::waitKey(0);
    fclose(file);
}

void rectOnly(std::string imageName)
{
    FILE * file = fopen("rects.txt", "r");
    std::vector<cv::Rect> rects;
    cv::Rect temp;
    while (fscanf(file, "%d%d%d%d", &temp.x, &temp.y, &temp.width, &temp.height) != EOF)
        rects.push_back(temp);
    cv::Mat result = cv::imread("C:\\GitHubCode\\anotovanie\\" + imageName);
    auto resultBoundingBoxes = nonMaxSuppression(rects, 0.5);
    for (cv::Rect& box : resultBoundingBoxes)
    {
        rectangle(result, box, (0, 0, 255), 2);
    }
    cv::imwrite("trieska.png", result);
    cv::waitKey(0);
    fclose(file);
}

int main(int argc, char* argv[])
{
    /*std::thread thread1 = std::thread(trainint, false, "trainedBoostNoBackfit.xml");
    trainint(true, "trainedBoost.xml");
    thread1.join();*/
    //trainint(true, "trainedBoostBackfit2.xml");
    //detect(true);
    /*std::thread thread1 = std::thread(detectMultiScale, false, "trainedBoost.xml", "outputBackfit.png");
    detectMultiScale(false, "trainedBoostNoBackfit.xml", "outputNoBackfit.png");
    thread1.join();*/
    rectOnly("SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_001975.png");
    //detectMultiScale(false, "trainedBoostBackfit2.xml", "outputBackfit2.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_001768.png");
    //detectMultiScale(false, "trainedBoostBackfit1.xml", "outputEmptyBackfitSupression.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_001975.png");
    //trainint();
    //Parser parser;
    //parser.parseNegatives();
    //parser.parsePositives();
    return 0;
}