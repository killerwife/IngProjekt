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
    char* __restrict imageData = (char*)img.data;
    const int cols = img.cols, rows = img.rows;
    float* gradient = new float[rows*cols * BINS];
    memset(gradient, 0, sizeof(float)*rows*cols * BINS);
    //#pragma loop(hint_parallel(8))
    //#pragma loop(ivdep)
    for (long i = 1; i < rows - 1; i++)
    {
        for (long k = 1; k < cols - 1; k++)
        {
            int dx = -imageData[i*cols + k - 1] + imageData[i*cols + k + 1];
            int dy = -imageData[(i - 1)*cols + k] + imageData[(i + 1)*cols + k];
            int absol1 = std::abs(dy);
            int absol2 = std::abs(dx);
            int orientation = (dy < 0) * 4 + (dx < 0) * 2 + (absol1 > absol2) * 1;
            gradient[rows*cols*orientation + i*cols + k] = float(absol1 + absol2);
        }
    }
    cv::Mat imageChannels[BINS] = {
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 0]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 1]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 2]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 3]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 4]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 5]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 6]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 7]),
    };
    double* integral = new double[(rows + 1)*(cols + 1) * 8];
    cv::Mat integralChannels[BINS] = {
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 0]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 1]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 2]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 3]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 4]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 5]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 6]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 7]),
    };
    tbb::parallel_for(size_t(0), size_t(7), [&](size_t i) { cv::integral(imageChannels[i], integralChannels[i]); });
    delete gradient;
    const int sizeRows = CELL_SIDE, sizeCols = CELL_SIDE;
    const int stepRows = STEP_SIZE, stepCols = STEP_SIZE;
    int* memory = histogram[idx].data();
    tbb::parallel_for(size_t(0), size_t(7), [&](size_t l) {
        for (int i = 0; i < rows - sizeRows; i += stepRows)
        {
            for (int k = 0; k < cols - sizeCols; k += stepCols)
            {
                memory[l*histSize + i / stepRows*(cols - sizeCols) + k / stepCols] =
                    int(integral[(l*(rows + 1) + i)*(cols + 1) + k] + integral[(l*(rows + 1) + i + sizeRows)*(cols + 1) + k + sizeCols]
                        - integral[(l*(rows + 1) + i)*(cols + 1) + k + sizeCols] - integral[(l*(rows + 1) + i + sizeRows)*(cols + 1) + k]);
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
