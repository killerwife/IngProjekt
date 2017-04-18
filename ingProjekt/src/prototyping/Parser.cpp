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
    std::vector<std::string> filenames;
    std::string path = "C:\\GitHubCode\\anotovanie\\TrainingData";
    GetFileNames(path, filenames);
    std::ofstream myfile;
    myfile.open("positives.txt");
    for (std::string& filename : filenames)
        myfile << "C:\\GitHubCode\\anotovanie\\TrainingData\\*.*" << filename << "\n";
    myfile.close();
}

void Parser::parsePositives()
{
    std::vector<std::string> filenames;
    std::string path = "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Training\\hrac\\RealData";
    GetFileNames(path, filenames);
    std::ofstream myfile;
    myfile.open("positives.txt");
    for (std::string& filename : filenames)
        myfile << "C:\\GitHubCode\\anotovanie\\BoundingBoxes\\Training\\hrac\\RealData\\" << filename << "\n";
    myfile.close();
}

void Parser::MakeDatFile(std::string& path, std::string& outputFile)
{
    std::vector<std::string> filenames;
    GetFileNames(path, filenames);
    std::ofstream myfile;
    myfile.open(outputFile);
    for (std::string& filename : filenames)
        myfile << path << "\\" << filename << std::endl;
    myfile.close();
}

void Parser::GetFileNames(std::string& path, std::vector<std::string>& filenames)
{
    std::string temp = path + "\\*.*";
    std::wstring stemp = std::wstring(temp.begin(), temp.end());
    LPCWSTR sw = stemp.c_str();
    HANDLE hFind;
    WIN32_FIND_DATA data;
    hFind = FindFirstFile(sw, &data);
    int i = 0;
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (i >= 2) // skip . and ..
            {
                stemp = data.cFileName;
                filenames.push_back(std::string(stemp.begin(), stemp.end()));
            }
            i++;
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
}

void Parser::toMat(cv::Mat& output, cv::Mat& responses, std::string location, int count, cv::Mat response)
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
            image = cv::imread((location + tempPosition).data(), CV_LOAD_IMAGE_GRAYSCALE);
            cv::imshow("bla", image);
            cv::waitKey(0);
            if (image.isContinuous())
            {
                image.convertTo(image, CV_32F);
                image = image.reshape(1, 1);
                output.push_back(image);
                responses.push_back(response);
            }
            i++;
            if (count <= i)
                break;
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
}

void Parser::toMat(std::vector<cv::Mat>& output, cv::Mat& responses, std::string location, int count, cv::Mat response)
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
                //cv::imshow("bla", image);
                //cv::waitKey(0);
                //image.convertTo(image, CV_32F);
                output.push_back(image);
                responses.push_back(response);
                //printf("%s\n", (location + tempPosition).data());
                //wprintf(L"%s\n", stemp.data());
                i++;
            }
            if (count <= i)
                break;
        } while (FindNextFile(hFind, &data));
        FindClose(hFind);
    }
}

