#include "SHOGfeatures.h"
#include "../cascadeclassifier.h"
#include "opencv2/core.hpp"

SHOGEvaluator::~SHOGEvaluator()
{
}

void SHOGEvaluator::init(const CvFeatureParams * _featureParams, int _maxSampleCount, cv::Size _winSize)
{
    histSize = (_winSize.height / STEP_SIZE - CELL_SIDE) * (_winSize.width / STEP_SIZE - CELL_SIDE);
    histCols = _winSize.width - CELL_SIDE;
    CvFeatureEvaluator::init(_featureParams, _maxSampleCount, _winSize);
}

void SHOGEvaluator::setImage(const cv::Mat & img, uchar clsLabel, int idx)
{

}

void SHOGEvaluator::writeFeatures(cv::FileStorage & fs, const cv::Mat & featureMap) const
{

}

void SHOGEvaluator::Feature::write(cv::FileStorage &fs) const
{

}

void SHOGEvaluator::generateFeatures()
{
    for (int i = 0; i < winSize.height; i += STEP_SIZE)
    {
        for (int k = 0; k < winSize.width; k += STEP_SIZE)
        {
            for (int n = 0; n < BINS; ++n)
            {
                features.push_back(Feature(i, k, n));
            }
        }
    }
}

CvSHOGFeatureParams::CvSHOGFeatureParams()
{
    name = SHOGFP_NAME;
}

SHOGEvaluator::Feature::Feature()
{
    data.x = 0;
    data.y = 0;
    data.ori = 0;
}

SHOGEvaluator::Feature::Feature(int x, int y, int orientation)
{
    data.x = x;
    data.y = y;
    data.ori = orientation;
}
