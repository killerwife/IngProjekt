#pragma once
#include "../traincascade_features.h"
#include <string>

#define SHOGFP_NAME "SHOGFeatureParams"
#define BINS 8
#define CELL_SIDE 4
#define STEP_SIZE 1

class CvSHOGFeatureParams : public CvFeatureParams
{
public:
    CvSHOGFeatureParams();
};

class SHOGEvaluator :
    public CvFeatureEvaluator
{
public:
    virtual ~SHOGEvaluator();
    virtual void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize);
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int varIdx, int sampleIdx) const;
    virtual void writeFeatures(cv::FileStorage &fs, const cv::Mat& featureMap) const;
protected:
    virtual void generateFeatures();

    class Feature
    {
    public:
        Feature();
        Feature(int x, int y, int orientation);
        void write(cv::FileStorage &fs) const;
        float calc(std::vector<int> hist, int step, int columns) const;
        
        struct
        {
            int x;
            int y;
            int ori;
        } data;
    };

    std::vector<Feature> features;

    std::vector<std::vector<int>> histogram;

    int histSize;
    int histCols;
};

float SHOGEvaluator::operator()(int varIdx, int sampleIdx) const
{
    return features[varIdx].calc(histogram[sampleIdx], histSize, histCols);
}

inline float SHOGEvaluator::Feature::calc(std::vector<int> hist, int step, int columns) const
{
    return hist[data.ori * step + data.x * columns + data.y];
}

