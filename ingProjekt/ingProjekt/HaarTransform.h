#pragma once
#include <opencv2/core.hpp>
#include "haarfeatures.h"
class HaarTransform
{
public:
	HaarTransform();
	~HaarTransform();

	void SetImages(std::vector<cv::Mat>& images);
	void GetFeature(std::vector<cv::Mat>& resultSet);

private:
	CvHaarEvaluator& m_eval;
};

