#include "SHOGfeatures.h"
#include "../cascadeclassifier.h"
#include "opencv2/core.hpp"
#include "tbb/tbb.h"
#include <opencv2/imgproc.hpp>

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
    CvFeatureEvaluator::setImage(img, clsLabel, idx);
    int* __restrict imageData = (int*)img.data;
    const int cols = img.cols, rows = img.rows;
    float* images = new float[rows*cols * 8];
    memset(images, 0, sizeof(float)*rows*cols * 8);
#pragma loop(hint_parallel(8))
#pragma loop(ivdep)
    for (long i = 1; i < rows - 1; i++)
    {
        for (long k = 1; k < cols - 1; k++)
        {
            int dx = -imageData[i*cols + k - 1] + imageData[i*cols + k + 1];
            int dy = -imageData[(i - 1)*cols + k] + imageData[(i + 1)*cols + k];
            int absol1 = std::abs(dy);
            int absol2 = std::abs(dx);
            int orientation = (dy < 0) * 4 + (dx < 0) * 2 + (absol1 > absol2) * 1;
            images[rows*cols*orientation + i*cols + k] = float(absol1 + absol2);
        }
    }
    cv::Mat imageChannels[8] = {
        cv::Mat(rows, cols, CV_32SC1, &images[rows*cols * 0]),
        cv::Mat(rows, cols, CV_32SC1, &images[rows*cols * 1]),
        cv::Mat(rows, cols, CV_32SC1, &images[rows*cols * 2]),
        cv::Mat(rows, cols, CV_32SC1, &images[rows*cols * 3]),
        cv::Mat(rows, cols, CV_32SC1, &images[rows*cols * 4]),
        cv::Mat(rows, cols, CV_32SC1, &images[rows*cols * 5]),
        cv::Mat(rows, cols, CV_32SC1, &images[rows*cols * 6]),
        cv::Mat(rows, cols, CV_32SC1, &images[rows*cols * 7]),
    };
    int* integral = new int[rows*cols * 8];
    cv::Mat integralChannels[8] = {
        cv::Mat(rows, cols, CV_32SC1, &integral[rows*cols * 0]),
        cv::Mat(rows, cols, CV_32SC1, &integral[rows*cols * 1]),
        cv::Mat(rows, cols, CV_32SC1, &integral[rows*cols * 2]),
        cv::Mat(rows, cols, CV_32SC1, &integral[rows*cols * 3]),
        cv::Mat(rows, cols, CV_32SC1, &integral[rows*cols * 4]),
        cv::Mat(rows, cols, CV_32SC1, &integral[rows*cols * 5]),
        cv::Mat(rows, cols, CV_32SC1, &integral[rows*cols * 6]),
        cv::Mat(rows, cols, CV_32SC1, &integral[rows*cols * 7]),
    };
    tbb::parallel_for(size_t(0), size_t(7), [&imageChannels, &integralChannels](size_t i) { cv::integral(imageChannels[i], integralChannels[i]); });
    delete images;
    const int sizeRows = CELL_SIDE, sizeCols = CELL_SIDE;
    const int stepRows = STEP_SIZE, stepCols = STEP_SIZE;
    histogram.push_back(std::vector<int>(histSize * 8));
    int* memory = histogram[histogram.size() - 1].data();
    tbb::parallel_for(size_t(0), size_t(7), [&](size_t l) {
        for (int i = 0; i < rows - sizeRows; i += stepRows)
        {
            for (int k = 0; k < cols - sizeCols; k += stepCols)
            {
                memory[l*histSize + i / stepRows*(cols - sizeCols) + k / stepCols] =
                    integral[(l*rows + i)*cols + k] + integral[(l*rows + i + sizeRows)*cols + k + sizeCols]
                    - integral[(l*rows + i)*cols + k + sizeCols] - integral[(l*rows + i + sizeRows)*cols + k];
            }
        }
    });
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
