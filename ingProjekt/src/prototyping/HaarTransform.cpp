#include "HaarTransform.h"
#include "opencv2/imgproc.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  


HaarTransform::HaarTransform(int maxSampleCount, cv::Size m_winSize) : m_sampleCount(maxSampleCount), m_counter(0)
{
	m_eval.init(CvFeatureParams::create(0), maxSampleCount, m_winSize);
}

HaarTransform::HaarTransform(HaarFeatureParameters mode, cv::Size winSize) : m_mode(mode), m_winSize(winSize)
{

}

HaarTransform::~HaarTransform()
{
}

void HaarTransform::SetImages(std::vector<cv::Mat>& images, cv::Mat& labels)
{
	for (int i = 0; i < images.size(); i++)
	{
		m_eval.setImage(images[i], labels.at<int>(i), i);
	}
}

void HaarTransform::SetImage(cv::Mat& image, int label)
{
	m_eval.setImage(image, label, m_counter);
}

void HaarTransform::GetFeatures(cv::Mat& resultSet)
{
	resultSet = cv::Mat();
	for (int i = 0; i < m_sampleCount; i++)
	{
		cv::Mat tempMat = cv::Mat(0, 0, CV_32F);
		for (int k = 0; k < m_eval.getNumFeatures(); k++)
		{
			tempMat.push_back(m_eval(k, i));
		}
		tempMat = tempMat.reshape(1, 1);
		resultSet.push_back(tempMat);
	}
}

void HaarTransform::SetImageBig(cv::Mat& image)
{
    m_image = image;
    int x = m_image.cols, y = m_image.rows;
    int count = 0;
    for (double scaleFactor = 1;;scaleFactor*=1.1,count++)
    {
        cv::Mat tempImage;
        cv::Size tempSize((int)std::round((double)x*(1. / scaleFactor)), (int)std::round((double)y*(1. / scaleFactor)));
        if (m_winSize.height > tempSize.height || m_winSize.width > tempSize.width)
            break;
        cv::resize(m_image,tempImage,tempSize);
        //cv::imshow("bla",tempImage);
        //cv::waitKey(0);
        m_sum.push_back(cv::Mat(0,0, m_image.type()));
        m_integral.push_back(cv::Mat(0, 0, m_image.type()));
        if (m_mode == HaarFeatureParameters::ALL)
        {
            m_tiltedIntegral.push_back(cv::Mat(0, 0, m_image.type()));
            cv::integral(tempImage, m_sum[count], m_integral[count], m_tiltedIntegral[count]);
        }
        else
            cv::integral(tempImage, m_sum[count], m_integral[count]);

        //cv::imshow("bla", tempImage);
        //cv::waitKey();
        //cv::imshow("bla", m_integral[count]);
        //cv::waitKey();
        //cv::imshow("bla", m_tiltedIntegral[count]);
        //cv::waitKey();

        generateFeatures(count,tempSize.width);
    }
}

void HaarTransform::CalculateFeatureVector(cv::Mat& features, int scale, int x, int y)
{
    float* data = (float*)features.data;
    for (int i = 0; i < m_features[scale].size(); i++)
    {
        data[i] = m_features[scale][i].calc(m_integral[scale], m_tiltedIntegral[scale], i, x, y);
    }
}

void HaarTransform::generateFeatures(int scale, int offset)
{
    m_features.push_back(std::vector<Feature>());
    for (int x = 0; x < m_winSize.width; x += 8)
    {
        for (int y = 0; y < m_winSize.height; y += 8)
        {
            for (int dx = 1; dx <= m_winSize.width; dx += 16)
            {
                for (int dy = 1; dy <= m_winSize.height; dy += 16)
                {
                    // haar_x2
                    if ((x + dx * 2 <= m_winSize.width) && (y + dy <= m_winSize.height))
                    {
                        m_features[scale].push_back(Feature(offset, false,
                            x, y, dx * 2, dy, -1,
                            x + dx, y, dx, dy, +2));
                    }
                    // haar_y2
                    if ((x + dx <= m_winSize.width) && (y + dy * 2 <= m_winSize.height))
                    {
                        m_features[scale].push_back(Feature(offset, false,
                            x, y, dx, dy * 2, -1,
                            x, y + dy, dx, dy, +2));
                    }
                    // haar_x3
                    if ((x + dx * 3 <= m_winSize.width) && (y + dy <= m_winSize.height))
                    {
                        m_features[scale].push_back(Feature(offset, false,
                            x, y, dx * 3, dy, -1,
                            x + dx, y, dx, dy, +3));
                    }
                    // haar_y3
                    if ((x + dx <= m_winSize.width) && (y + dy * 3 <= m_winSize.height))
                    {
                        m_features[scale].push_back(Feature(offset, false,
                            x, y, dx, dy * 3, -1,
                            x, y + dy, dx, dy, +3));
                    }
                    if (m_mode != CvHaarFeatureParams::BASIC)
                    {
                        // haar_x4
                        if ((x + dx * 4 <= m_winSize.width) && (y + dy <= m_winSize.height))
                        {
                            m_features[scale].push_back(Feature(offset, false,
                                x, y, dx * 4, dy, -1,
                                x + dx, y, dx * 2, dy, +2));
                        }
                        // haar_y4
                        if ((x + dx <= m_winSize.width) && (y + dy * 4 <= m_winSize.height))
                        {
                            m_features[scale].push_back(Feature(offset, false,
                                x, y, dx, dy * 4, -1,
                                x, y + dy, dx, dy * 2, +2));
                        }
                    }
                    // x2_y2
                    if ((x + dx * 2 <= m_winSize.width) && (y + dy * 2 <= m_winSize.height))
                    {
                        m_features[scale].push_back(Feature(offset, false,
                            x, y, dx * 2, dy * 2, -1,
                            x, y, dx, dy, +2,
                            x + dx, y + dy, dx, dy, +2));
                    }
                    if (m_mode != CvHaarFeatureParams::BASIC)
                    {
                        if ((x + dx * 3 <= m_winSize.width) && (y + dy * 3 <= m_winSize.height))
                        {
                            m_features[scale].push_back(Feature(offset, false,
                                x, y, dx * 3, dy * 3, -1,
                                x + dx, y + dy, dx, dy, +9));
                        }
                    }
                    if (m_mode == CvHaarFeatureParams::ALL)
                    {
                        // tilted haar_x2
                        if ((x + 2 * dx <= m_winSize.width) && (y + 2 * dx + dy <= m_winSize.height) && (x - dy >= 0))
                        {
                            m_features[scale].push_back(Feature(offset, true,
                                x, y, dx * 2, dy, -1,
                                x, y, dx, dy, +2));
                        }
                        // tilted haar_y2
                        if ((x + dx <= m_winSize.width) && (y + dx + 2 * dy <= m_winSize.height) && (x - 2 * dy >= 0))
                        {
                            m_features[scale].push_back(Feature(offset, true,
                                x, y, dx, 2 * dy, -1,
                                x, y, dx, dy, +2));
                        }
                        // tilted haar_x3
                        if ((x + 3 * dx <= m_winSize.width) && (y + 3 * dx + dy <= m_winSize.height) && (x - dy >= 0))
                        {
                            m_features[scale].push_back(Feature(offset, true,
                                x, y, dx * 3, dy, -1,
                                x + dx, y + dx, dx, dy, +3));
                        }
                        // tilted haar_y3
                        if ((x + dx <= m_winSize.width) && (y + dx + 3 * dy <= m_winSize.height) && (x - 3 * dy >= 0))
                        {
                            m_features[scale].push_back(Feature(offset, true,
                                x, y, dx, 3 * dy, -1,
                                x - dy, y + dy, dx, dy, +3));
                        }
                        // tilted haar_x4
                        if ((x + 4 * dx <= m_winSize.width) && (y + 4 * dx + dy <= m_winSize.height) && (x - dy >= 0))
                        {
                            m_features[scale].push_back(Feature(offset, true,
                                x, y, dx * 4, dy, -1,
                                x + dx, y + dx, dx * 2, dy, +2));
                        }
                        // tilted haar_y4
                        if ((x + dx <= m_winSize.width) && (y + dx + 4 * dy <= m_winSize.height) && (x - 4 * dy >= 0))
                        {
                            m_features[scale].push_back(Feature(offset, true,
                                x, y, dx, 4 * dy, -1,
                                x - dy, y + dy, dx, 2 * dy, +2));
                        }
                    }
                }
            }
        }
    }
}

float Feature::calc(const cv::Mat &_sum, const cv::Mat &_tilted, size_t y, size_t offsetX, size_t offsetY) const
{
    const int* img = (m_tilted ? _tilted.ptr<int>() : _sum.ptr<int>());
    size_t combinedOffset = offsetX + m_offset*offsetY;
    int firstFirst = img[fastRect[0].p0 + combinedOffset];
    int firstSecond = img[fastRect[0].p1 + combinedOffset];
    int firstThird = img[fastRect[0].p2 + combinedOffset];
    int firstFourth = img[fastRect[0].p3 + combinedOffset];
    int firstRect = firstFirst - firstSecond - firstThird + firstFourth;
    int secondFirst = img[fastRect[1].p0 + combinedOffset];
    int secondSecond = img[fastRect[1].p1 + combinedOffset];
    int secondThird = img[fastRect[1].p2 + combinedOffset];
    int secondFourth = img[fastRect[1].p3 + combinedOffset];
    int secondRect = secondFirst - secondSecond - secondThird + secondFourth;
    float ret = rect[0].weight * (firstRect) +
        rect[1].weight * (secondRect);
    if (rect[2].weight != 0.0f)
        ret += rect[2].weight * (img[fastRect[2].p0 + combinedOffset] - img[fastRect[2].p1 + combinedOffset] - img[fastRect[2].p2 + combinedOffset] + img[fastRect[2].p3 + combinedOffset]);
    return ret;
}

Feature::Feature() : m_tilted(false)
{
    rect[0].r = rect[1].r = rect[2].r = cv::Rect(0, 0, 0, 0);
    rect[0].weight = rect[1].weight = rect[2].weight = 0;
}

Feature::Feature(int offset, bool _tilted,
    int x0, int y0, int w0, int h0, float wt0,
    int x1, int y1, int w1, int h1, float wt1,
    int x2, int y2, int w2, int h2, float wt2) : m_offset(offset), m_tilted(_tilted)
{
    rect[0].r.x = x0;
    rect[0].r.y = y0;
    rect[0].r.width = w0;
    rect[0].r.height = h0;
    rect[0].weight = wt0;

    rect[1].r.x = x1;
    rect[1].r.y = y1;
    rect[1].r.width = w1;
    rect[1].r.height = h1;
    rect[1].weight = wt1;

    rect[2].r.x = x2;
    rect[2].r.y = y2;
    rect[2].r.width = w2;
    rect[2].r.height = h2;
    rect[2].weight = wt2;

    if (!m_tilted)
    {
        for (int j = 0; j < CV_HAAR_FEATURE_MAX; j++)
        {
            if (rect[j].weight == 0.0F)
                break;
            CV_SUM_OFFSETS(fastRect[j].p0, fastRect[j].p1, fastRect[j].p2, fastRect[j].p3, rect[j].r, m_offset)
        }
    }
    else
    {
        for (int j = 0; j < CV_HAAR_FEATURE_MAX; j++)
        {
            if (rect[j].weight == 0.0F)
                break;
            CV_TILTED_OFFSETS(fastRect[j].p0, fastRect[j].p1, fastRect[j].p2, fastRect[j].p3, rect[j].r, m_offset)
        }
    }
}
