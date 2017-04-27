#include "SHOGfeaturesTrain.h"
#include "../cascadeclassifier.h"
#include "opencv2/core.hpp"
#include "tbb/tbb.h"
#include <opencv2/imgproc.hpp>
#include "../../SharedDefines/SharedDefines.h"

using namespace train;

SHOGEvaluator::~SHOGEvaluator()
{
}

void SHOGEvaluator::init(const CvFeatureParams * _featureParams, int _maxSampleCount, cv::Size _winSize)
{
    histSize = (_winSize.height / STEP_SIZE - CELL_SIDE) * (_winSize.width / STEP_SIZE - CELL_SIDE);
    histCols = _winSize.width / STEP_SIZE - CELL_SIDE;
    histRows = _winSize.height / STEP_SIZE - CELL_SIDE;
    histogram.resize(_maxSampleCount);
    printf("Histogram array size: %d histSize %d\n", _maxSampleCount, histSize);
    for (auto& vector : histogram)
        vector.resize(histSize * BINS);
    CvFeatureEvaluator::init(_featureParams, _maxSampleCount, _winSize);
}

void SHOGEvaluator::setImage(const cv::Mat & img, uchar clsLabel, int idx)
{
    CvFeatureEvaluator::setImage(img, clsLabel, idx);
    const int cols = img.cols, rows = img.rows;
    float* gradient = new float[rows*cols * BINS];
    double* integral = new double[(rows + 1)*(cols + 1) * BINS];
    ComputeSHOG(img, gradient, integral, histogram[idx].data(), BINS, cv::Size(STEP_SIZE, STEP_SIZE), cv::Size(CELL_SIDE, CELL_SIDE));
    delete gradient;
    delete integral;
}

void SHOGEvaluator::writeFeatures(cv::FileStorage & fs, const cv::Mat & featureMap) const
{
    _writeFeatures(features, fs, featureMap);
}

void SHOGEvaluator::Feature::write(cv::FileStorage &fs) const
{
    fs << S_HOG << "[" << data.x << data.y << data.ori << "]";
}

void SHOGEvaluator::generateFeatures()
{
    for (int i = 0; i < histRows; i++)
    {
        for (int k = 0; k < histCols; k++)
        {
            for (int n = 0; n < BINS; ++n)
            {
                features.push_back(Feature(i, k, n));
            }
        }
    }
    numFeatures = (int)features.size();
    printf("Number of features generated: %d\n", numFeatures);
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
