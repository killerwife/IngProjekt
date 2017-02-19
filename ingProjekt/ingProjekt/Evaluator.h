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
#include "HaarTransform.h"

class Evaluator
{
public:
    Evaluator();
    ~Evaluator();

    void fillData(cv::Mat& data, cv::Mat& responses, bool backfitting, int countPos, int countNeg, std::string sampleFolders[3], int countBackfit = 0);
    void fillData(std::vector<cv::Mat>& data, cv::Mat& responses, bool backfitting, int countPos, int countNeg, std::string sampleFolders[3], int countBackfit = 0);

    void trainintRawImage(bool backfitting, std::string xml, std::string sampleFolders[3]);
	void detectRawImage(std::string filename, bool backfitting, std::string sampleFolders[3]);
    void trainint(bool backfitting, std::string xml, std::string sampleFolders[3]);
	void detect(bool backfitting, std::string filename, std::string sampleFolders[3]);
	void detectMultiScale(bool exportShit, std::string xml, std::string filename, std::string imageName);
	void detectMultiScaleTemp(bool exportShit, std::string xml, std::string filename, std::string imageName);
    void detectMultiScaleProto(bool exportShit, std::string xml, std::string filename, std::string imageName);
};

