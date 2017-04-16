#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  
#include <opencv2/core/cuda.hpp> 
#include <opencv2/objdetect.hpp>
#include <opencv2/cudaobjdetect.hpp>

class FeaturePrototypes
{
public:
    FeaturePrototypes();
    ~FeaturePrototypes();

    void ComputeSHog(cv::Mat & image);
    void ComputeHistFeat(cv::Mat & image);

    static void test();
};

