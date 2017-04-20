#pragma once
#include "../cascadedetect.hpp"

class SHOGEvaluator :
    public cv::FeatureEvaluator
{
public:
    struct Feature
    {

    };

    struct OptFeature
    {

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
        return 0.0f;// optfeaturesPtr[featureIdx].calc(pwin) * varianceNormFactor;
    }
    virtual float calcOrd(int featureIdx) const
    {
        return (*this)(featureIdx);
    }

protected:
    virtual void computeChannels(int i, cv::InputArray img);
    virtual void computeOptFeatures();

    std::vector<Feature> features;
    std::vector<OptFeature> optfeatures;
    std::vector<OptFeature> optfeatures_lbuf;
};

