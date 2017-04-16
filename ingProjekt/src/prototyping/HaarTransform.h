#pragma once
#include <opencv2/core.hpp>
#include "haarfeatures.h"

enum HaarFeatureParameters { BASIC = 0, CORE = 1, ALL = 2 };

class Feature
{
public:
    Feature();
    Feature(int offset, bool _tilted,
        int x0, int y0, int w0, int h0, float wt0,
        int x1, int y1, int w1, int h1, float wt1,
        int x2 = 0, int y2 = 0, int w2 = 0, int h2 = 0, float wt2 = 0.0F);
    float calc(const cv::Mat &sum, const cv::Mat &tilted, size_t y, size_t offsetX, size_t offsetY) const;

    bool m_tilted;
    struct
    {
        cv::Rect r;
        float weight;
    } rect[CV_HAAR_FEATURE_MAX];

    struct
    {
        int p0, p1, p2, p3;
    } fastRect[CV_HAAR_FEATURE_MAX];

    int m_offset;
};

class HaarTransform
{
public:
    HaarTransform(int maxSampleCount, cv::Size winsize);
    HaarTransform(HaarFeatureParameters mode, cv::Size winSize);
	~HaarTransform();

    void SetImages(std::vector<cv::Mat>& images, cv::Mat& labels);
    void GetFeatures(cv::Mat& resultSet);
	void SetImage(cv::Mat& image, int label);
    void SetImageBig(cv::Mat& image);
    void CalculateFeatureVector(cv::Mat& features, int scale, int x, int y);
    void generateFeatures(int scale, int offset);
    size_t GetFeatureCount() { return m_features[0].size(); }
private:
	CvHaarEvaluator m_eval;
    int m_sampleCount;
	int m_counter;

    cv::Mat m_image;
    std::vector<cv::Mat> m_sum;
    std::vector<cv::Mat> m_integral;
    std::vector<cv::Mat> m_tiltedIntegral;
    std::vector<std::vector<Feature>> m_features;
    HaarFeatureParameters m_mode;
    cv::Size m_winSize;
};

