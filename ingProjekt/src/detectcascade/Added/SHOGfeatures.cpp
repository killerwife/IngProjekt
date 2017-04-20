#include "SHOGfeatures.h"
#include "opencv2/core.hpp"

SHOGEvaluator::SHOGEvaluator()
{
}

SHOGEvaluator::~SHOGEvaluator()
{
}

bool SHOGEvaluator::read(const cv::FileNode & node, cv::Size origWinSize)
{
    return false;
}

cv::Ptr<cv::FeatureEvaluator> SHOGEvaluator::clone() const
{
    return cv::Ptr<cv::FeatureEvaluator>();
}

bool SHOGEvaluator::setWindow(cv::Point p, int scaleIdx)
{
    return false;
}

cv::Rect SHOGEvaluator::getNormRect() const
{
    return cv::Rect();
}

int SHOGEvaluator::getSquaresOffset() const
{
    return 0;
}

void SHOGEvaluator::computeChannels(int i, cv::InputArray img)
{
}

void SHOGEvaluator::computeOptFeatures()
{
}
