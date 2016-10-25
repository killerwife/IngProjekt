#pragma once
#include <opencv2/core.hpp>
class HaarTransform
{
public:
	HaarTransform();
	~HaarTransform();

	float GetFeature(cv::Mat &img, cv::Mat &mask);
};

