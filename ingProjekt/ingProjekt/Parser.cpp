#include "Parser.h"
#include <windows.h>
#include <fstream>
#include <string>

Parser::Parser()
{
}


Parser::~Parser()
{
}

void Parser::parseNegatives()
{
    std::string temp = "C:\\GitHubCode\\anotovanie\\TrainingData\\*.*";
    std::wstring stemp = std::wstring(temp.begin(), temp.end());
    std::ofstream myfile;
    myfile.open("negatives.txt");
    LPCWSTR sw = stemp.c_str();
    HANDLE hFind;
    WIN32_FIND_DATA data;
    hFind = FindFirstFile(sw, &data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            stemp = data.cFileName;
            temp = std::string(stemp.begin(), stemp.end());
            //std::cout << temp << "\n";
            myfile << "C:\\GitHubCode\\anotovanie\\TrainingData\\" << temp << "\n";
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
    myfile.close();
}

void Parser::parsePositives()
{
    std::string temp = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Training\\hrac\\RealData\\*.*";
    std::wstring stemp = std::wstring(temp.begin(), temp.end());
    std::ofstream myfile;
    myfile.open("positives.txt");
    LPCWSTR sw = stemp.c_str();
    HANDLE hFind;
    WIN32_FIND_DATA data;
    hFind = FindFirstFile(sw, &data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            stemp = data.cFileName;
            temp = std::string(stemp.begin(), stemp.end());
            //std::cout << temp << "\n";
            myfile << "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Training\\hrac\\RealData\\" << temp << "\n";
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
    myfile.close();
}

void Parser::toMat(cv::Mat** output, cv::Mat** responses)
{
    *output = new cv::Mat(0,0,CV_32F);
    *responses = new cv::Mat(0, 0, CV_32F);
    int positive=22, negative=22;
    int i = 0;
    cv::Mat image;
    float arrayPos[1] = {1};
    cv::Mat pos(1, 1, CV_32F,arrayPos);
    float arrayNeg[1] = { 0 };
    cv::Mat neg(1, 1, CV_32F, arrayNeg);
    std::string tempPos = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Training\\hrac\\RealData\\*.*";
    std::string tempNeg = "C:\\GitHubCode\\anotovanie\\TrainingData\\*.*";
    std::wstring stemp = std::wstring(tempPos.begin(), tempPos.end());
    LPCWSTR sw = stemp.c_str();
    HANDLE hFind;
    WIN32_FIND_DATA data;
    hFind = FindFirstFile(sw, &data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            stemp = data.cFileName;
            tempPos = std::string(stemp.begin(), stemp.end());
            //std::cout << temp << "\n";
            image = cv::imread("C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Training\\hrac\\RealData\\"+tempPos, CV_LOAD_IMAGE_GRAYSCALE);
            if (image.isContinuous())
            {
                image.convertTo(image, CV_32F);
                image=image.reshape(1, 1);
                (*output)->push_back(image);
                (*responses)->push_back(pos);
            }
            i++;
            if (positive <= i)
                break;
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
    i = 0;
    stemp = std::wstring(tempNeg.begin(), tempNeg.end());
    sw = stemp.c_str();
    hFind = FindFirstFile(sw, &data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            stemp = data.cFileName;
            tempNeg = std::string(stemp.begin(), stemp.end());
            //std::cout << temp << "\n";
            image = cv::imread("C:\\GitHubCode\\anotovanie\\TrainingData\\"+tempNeg, CV_LOAD_IMAGE_GRAYSCALE);
            if (image.isContinuous())
            {
                image.convertTo(image, CV_32F);
                image=image.reshape(1, 1);
                (*output)->push_back(image);
                (*responses)->push_back(neg);
            }
            i++;
            if (negative <= i)
                break;
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
}

