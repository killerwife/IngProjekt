#pragma once
#include "../cascadedetect.hpp"
#include <memory>

#define BINS 8
#define CELL_SIDE 4
#define STEP_SIZE 1
#define S_HOG "shog"

class SHOGEvaluator : public cv::FeatureEvaluator
{
public:
    struct Feature
    {
        Feature();
        bool read(const cv::FileNode& node);

        struct
        {
            int x;
            int y;
            int ori;
        } data;
    };

    struct OptFeature
    {
        OptFeature() : offset(0) {}

        float calc(const int* pwin) const;
        void setOffset(const Feature& _f, int histsize, int columns);

        int offset;
    };

    SHOGEvaluator();
    virtual ~SHOGEvaluator();

    virtual bool read(const cv::FileNode& node, cv::Size origWinSize);
    virtual cv::Ptr<cv::FeatureEvaluator> clone() const;
    virtual int getFeatureType() const { return FeatureEvaluator::SHOG; }

    virtual bool setWindow(cv::Point p, int scaleIdx);
    cv::Rect getNormRect() const;
    int getSquaresOffset() const;

    float operator()(int featureIdx) const
    {
        return optfeatures[featureIdx].calc(&(*histogramData)[curScaleIdx][winStartOffset]);
    }
    virtual float calcOrd(int featureIdx) const
    {
        return (*this)(featureIdx);
    }

protected:
    virtual void computeChannels(int i, cv::InputArray img);
    virtual void computeOptFeatures();

    std::shared_ptr<std::vector<Feature>> features;
    std::vector<OptFeature> optfeatures;

    OptFeature* optfeaturesPtr;

    std::shared_ptr<std::vector<std::vector<int>>> histogramData;
    std::shared_ptr<std::vector<float>> gradient;
    std::shared_ptr<std::vector<double>> integral;

    size_t winStartOffset;
    const int* winStart;

    struct HistData
    {
        int histSize;
        int histCols;
        int histRows;
    };

    std::shared_ptr<std::vector<HistData>> histogramSizes;

    int curScaleIdx;
};

inline float SHOGEvaluator::OptFeature::calc(const int* pwin) const
{
    return float(pwin[offset]);
}

inline void SHOGEvaluator::OptFeature::setOffset(const Feature& _f, int histsize, int columns)
{
    offset = _f.data.ori * histsize + _f.data.x * columns + _f.data.y;
}

