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
    double TakeTime;
    unsigned long long Atime, Btime;

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

/*
Fills data matrices with the following:
data is filled with one row per picture, responses is filed with one response value per picture
backfitting specifies whether a secondary false positive folder should be used
countPos, countNeg and countBackfit specify amounts of each respective cathegory
sampleFolders[0] specifies folder for positive samples
sampleFolders[1] specifies folder for negative samples
sampleFolders[2] specifies folder for backfitting samples
*/
void fillData(cv::Mat** data, cv::Mat** responses, bool backfitting, int countPos, int countNeg, std::string sampleFolders[3], int countBackfit = 0)
{
    Parser parser;
    *data = new cv::Mat(0, 0, CV_32S);
    *responses = new cv::Mat(0, 0, CV_32S);
    int arrayPos[1] = { 0 };
    cv::Mat pos(1, 1, CV_32S, arrayPos);
    parser.toMat(data, responses, sampleFolders[0], countPos, pos);
    int arrayNeg[1] = { 1 };
    cv::Mat neg(1, 1, CV_32S, arrayNeg);
    parser.toMat(data, responses, sampleFolders[1], countNeg, neg);
    if (backfitting)
        parser.toMat(data, responses, sampleFolders[2], countBackfit, neg);
}

/*
Trains an XML with sample data.
Backfitting specifies whether a secondary false negative folder should be used.
sampleFolders[0] specifies folder for positive samples
sampleFolders[1] specifies folder for negative samples
sampleFolders[2] specifies folder for backfitting samples
*/
void trainint(bool backfitting, std::string xml, std::string sampleFolders[3])
{
    cv::Mat* data = nullptr;
    cv::Mat* responses = nullptr;
    fillData(&data, &responses, backfitting, 5000, 20000, sampleFolders, 8000);
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

void detect(std::string filename, bool backfitting, std::string sampleFolders[3])
{
    Parser parser;
    cv::Mat* data = nullptr;
    cv::Mat* responses = nullptr;
    fillData(&data, &responses, backfitting, 40000, 40000, sampleFolders, 40000);
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

/*
Comparator for sorting cv::Rect according to our needs
Currently unused
*/
struct RectComparator
{
    bool operator()(cv::Rect a, cv::Rect b)
    {
        return (a.y + a.height) < (b.y + b.height);
    }
};

/*
Performs flavor of non-maximum-suppression upon boundingBoxes, which contains all detected rectangles.
Overlap threshold is usually set to between 0.3-0.5.
Multiplier specifies the maximum area difference two bounding boxes can have to still be joined together
Multiplier 4 generally means a bounding box 2x the size
*/
std::vector<cv::Rect> nonMaxSuppression(std::vector<cv::Rect> boundingBoxes, float overlapThreshold, int multiplier)
{
    std::vector<cv::Rect> result;
    //std::sort(boundingBoxes.begin(), boundingBoxes.end(), RectComparator());
    /*for (cv::Rect rect : boundingBoxes)
        std::cout << rect << "\t";*/
    std::vector<double> areas;
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
    //std::vector<int> candidates;
    for (int i = 0; i < boundingBoxes.size();)
    {
        int candidate = -1;
        for (int j = 0; j < boundingBoxes.size(); j++)
        {
            if (i == j || (x[i]>x[j] + widths[j] || x[i] + widths[i]<x[j]) || (y[i]>y[j] + heights[j] || y[i] + heights[i] < y[j]))
                continue;

            int xx1 = x[i] > x[j] ? x[i] : x[j];
            int yy1 = y[i] > y[j] ? y[i] : y[j];
            int xx2 = x[i] + widths[i] < x[j] + widths[j] ? x[i] + widths[i] : x[j] + widths[j];
            int yy2 = y[i] + heights[i] < x[j] + heights[j] ? x[i] + heights[i] : x[j] + heights[j];
            double w = xx2 - xx1 + 1 > 0 ? xx2 - xx1 + 1 : 0;
            double h = yy2 - yy1 + 1 > 0 ? yy2 - yy1 + 1 : 0;
            double overlap = w*h / (areas[j]+areas[i]);
            double areaDiff = areas[i] / areas[j];
            if (overlap > overlapThreshold && (areaDiff > 1.f / multiplier && areaDiff < multiplier))
            {
                //candidates.push_back(j);
                candidate = j;
                break;
            }
            //printf("%d\t", j);
        }
        //printf("\n%d\n", i);
        if (candidate != -1) // TODO: add joining of several bounding boxes at once
        {
            int height = (heights[i] + heights[candidate]) / 2;
            int width = (widths[i] + widths[candidate]) / 2;
            int finalWeight = weights[candidate] + weights[i];
            int middleFirstX = x[i] + widths[i] / 2;
            int middleSecondX = ((x[candidate] + widths[candidate]) / 2);
            int middleFirstY = (y[i] + heights[i]) / 2;
            int middleSecondY = ((y[candidate] + heights[candidate]) / 2);
            int middleX = (int)((double)abs(middleFirstX - middleSecondX) / finalWeight*weights[candidate]);
            int middleY = (int)((double)abs(middleFirstY - middleSecondY) / finalWeight*weights[candidate]);
            int coordX = middleX - width / 2 + x[i]>x[candidate] ? x[candidate] : x[i];
            int coordY = middleY - height / 2 + y[i] > y[candidate] ? y[candidate] : y[i];
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

/*
Multiscale detection using a trained boost model.
exportShit specifies whether found detection regions should be exported into images on the hard drive (used for backfitting)
filename is the path used for saving final detection result
imageName is the image upon which we want to launch the detection algorithm
*/
void detectMultiScale(bool exportShit, std::string xml, std::string filename, std::string imageName)
{
    cv::Ptr<cv::ml::Boost> boost = cv::Algorithm::load<cv::ml::Boost>(xml);
    cv::Mat img = cv::imread("C:\\GitHubCode\\anotovanie\\" + imageName);
    //cv::Mat img = cv::imread("C:\\GitHubCode\\IngProjekt\\ingProjekt\\ingProjekt\\" + imageName);
    cv::Mat result = img.clone();
    std::vector<cv::Rect> boundingBoxes;
    FILE * file = fopen("rects.txt", "w");
    int shift = 4;
    int m = 32000;
    for (int scale = 8; scale <= 512; scale *= 1.25f, shift *= 1.25f)
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
    auto resultBoundingBoxes = nonMaxSuppression(boundingBoxes, 0.3f, 4);
    for (cv::Rect& box : resultBoundingBoxes)
    {
        rectangle(result, box, (0, 0, 255), 2);
    }
    cv::imwrite(filename, result);
    cv::waitKey(0);
    fclose(file);
}

/*
Function for non-maximum-suppression testing.
Enables quick loading of rectangles from file instead of requiring a detection algorithm run.
*/
void rectOnly(std::string imageName)
{
    FILE * file = fopen("rects.txt", "r");
    std::vector<cv::Rect> rects;
    cv::Rect temp;
    while (fscanf(file, "%d%d%d%d", &temp.x, &temp.y, &temp.width, &temp.height) != EOF)
        rects.push_back(temp);
    cv::Mat result = cv::imread("C:\\GitHubCode\\anotovanie\\" + imageName);
    //cv::imshow("bla", result);
    //cv::waitKey(0);
    auto resultBoundingBoxes = nonMaxSuppression(rects, 0.3, 4);
    for (cv::Rect& box : resultBoundingBoxes)
    {
        rectangle(result, box, (0, 0, 255), 2);
    }
    cv::imwrite("trieska2.png", result);
    cv::waitKey(0);
    fclose(file);
}

/*
ocasovat cpu boost a gpu boost, trening a detekcia niekolkych obrazkov
*/

int main(int argc, char* argv[])
{
    /*std::thread thread1 = std::thread(trainint, false, "trainedBoostNoBackfit.xml");
    trainint(true, "trainedBoost.xml");
    thread1.join();*/
    std::string sampleFolders[3];
    sampleFolders[0] = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Testing\\hrac\\RealData\\";
    sampleFolders[1] = "C:\\GitHubCode\\anotovanie\\TrainingData\\";
    sampleFolders[2] = "C:\\GitHubCode\\backfitting\\";
    //trainint(true, "trainedBoostFinal3.xml",sampleFolders);
    //detect(true);
    /*std::thread thread1 = std::thread(detectMultiScale, false, "trainedBoost.xml", "outputBackfit.png");
    detectMultiScale(false, "trainedBoostNoBackfit.xml", "outputNoBackfit.png");
    thread1.join();*/
    /*
    std::string sampleFolders[3];
    sampleFolders[0] = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Testing\\hrac\\RealData\\";
    sampleFolders[1] = "C:\\GitHubCode\\anotovanie\\TrainingData\\";
    sampleFolders[2] = "C:\\GitHubCode\\backfitting\\";
    */
    rectOnly("SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002000.png");
    //detectMultiScale(true, "trainedBoostFinal2.xml", "outputFinal2.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_001975.png");
    //detectMultiScale(false, "trainedBoostFinal0.xml", "outputFinalNotBackfitted1.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002000.png");
    //detectMultiScale(false, "trainedBoostFinal3.xml", "outputFinalBackfitted1.png", "SNO-7084R_192.168.1.100_80-Cam01_H.264_2048X1536_fps_30_20151115_202619.avi_2fps_002000.png");
    //trainint();
    //Parser parser;
    //parser.parseNegatives();
    //parser.parsePositives();
    return 0;
}