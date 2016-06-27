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

void Parser::toMat(cv::Mat** output, cv::Mat** responses, std::string location, int count, cv::Mat response)
{
    int i = 0;
    cv::Mat image;
    std::string start = location + "*.*";
	std::wstring stemp = std::wstring(start.begin(), start.end());
    LPCWSTR sw = stemp.c_str();
    HANDLE hFind;
    WIN32_FIND_DATA data;
    hFind = FindFirstFile(sw, &data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            stemp = data.cFileName;
            std::string tempPosition = std::string(stemp.begin(), stemp.end());
            //std::cout << temp << "\n";
            image = cv::imread(location + tempPosition, CV_LOAD_IMAGE_GRAYSCALE);
            if (image.isContinuous())
            {
                image.convertTo(image, CV_32F);
                image=image.reshape(1, 1);
                (*output)->push_back(image);
                (*responses)->push_back(response);
            }
            i++;
            if (count <= i)
                break;
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
}

