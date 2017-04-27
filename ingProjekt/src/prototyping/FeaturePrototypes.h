#pragma once
#include <opencv2/core.hpp>

class FeaturePrototypes
{
public:
    FeaturePrototypes();
    ~FeaturePrototypes();

    void TestSHOG(cv::Mat & image);
    void ComputeHistFeat(cv::Mat & image);

    static void test();
};

