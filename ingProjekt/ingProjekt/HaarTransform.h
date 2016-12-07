#pragma once
#include <opencv2/core.hpp>
#include "haarfeatures.h"
class HaarTransform
{
public:
    HaarTransform(int maxSampleCount, cv::Size winsize);
	~HaarTransform();

    void SetImages(std::vector<cv::Mat>& images, cv::Mat& labels);
    void GetFeatures(cv::Mat& resultSet);
	void SetImage(cv::Mat& image, int label);
private:
	CvHaarEvaluator m_eval;
    int m_sampleCount;
	int m_counter;
};

