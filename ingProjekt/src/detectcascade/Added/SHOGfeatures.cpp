#include "SHOGfeatures.h"
#include "opencv2/core.hpp"
#include <tbb/tbb.h>
#include <opencv2/imgproc.hpp>
#include "../../SharedDefines/SharedDefines.h"

using namespace detect;

SHOGEvaluator::SHOGEvaluator()
{
}

SHOGEvaluator::~SHOGEvaluator()
{
    //printf("%d\n", 0);
}

SHOGEvaluator::Feature::Feature()
{
    data.ori = 0;
    data.x = 0;
    data.y = 0;
}

bool SHOGEvaluator::Feature::read(const cv::FileNode& node)
{
    cv::FileNode rnode = node[S_HOG];
    cv::FileNodeIterator it = rnode.begin(), it_end = rnode.end();
    it >> data.x >> data.y >> data.ori;
    return true;
}

bool SHOGEvaluator::read(const cv::FileNode & node, cv::Size origWinSize)
{
    if (!FeatureEvaluator::read(node, origWinSize))
        return false;
    if (!features)
        features = std::make_shared<std::vector<Feature>>();
    if (!gradient)
        gradient = std::make_shared<std::vector<float>>();
    if (!histogramData)
        histogramData = std::make_shared<std::vector<std::vector<int>>>();
    if (!integral)
        integral = std::make_shared<std::vector<double>>();
    if (!histogramSizes)
        histogramSizes = std::make_shared<std::vector<HistData>>();
    size_t i, n = node.size();
    CV_Assert(n > 0);
    features->resize(n);
    cv::FileNodeIterator it = node.begin();
    for (i = 0; i < n; i++, ++it)
    {
        if (!(*features)[i].read(*it))
            return false;
    }
    nchannels = 1;
    return true;
}

cv::Ptr<cv::FeatureEvaluator> SHOGEvaluator::clone() const
{
    cv::Ptr<SHOGEvaluator> ret = cv::makePtr<SHOGEvaluator>();
    *ret = *this;
    return ret;
}

bool SHOGEvaluator::setWindow(cv::Point pt, int scaleIdx)
{
    const ScaleData& s = getScaleData(scaleIdx);

    if (pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= s.szi.width ||
        pt.y + origWinSize.height >= s.szi.height)
        return false;

    if (curScaleIdx != scaleIdx)
    {
        curScaleIdx = scaleIdx;
        HistData& data = (*histogramSizes)[curScaleIdx];
        for (int i = 0; i < optfeatures.size(); i++)
        {
            optfeatures[i].setOffset((*features)[i], data.histSize, data.histCols);
            //printf("Feature %d scale ID %d new offset %d\n", i, curScaleIdx, optfeatures[i].offset);
        }
    }
    winStartOffset = pt.y * (*histogramSizes)[curScaleIdx].histCols + pt.x;
    return true;
}

// opencl function
cv::Rect SHOGEvaluator::getNormRect() const
{
    return cv::Rect();
}

// opencl function
int SHOGEvaluator::getSquaresOffset() const
{
    return 0;
}

void SHOGEvaluator::computeChannels(int i, cv::InputArray img)
{
    cv::Mat imageMat = img.getMat();
    ComputeSHOG(imageMat, this->gradient->data(), this->integral->data(), (*histogramData)[i].data(), BINS, cv::Size(STEP_SIZE, STEP_SIZE), cv::Size(CELL_SIDE, CELL_SIDE));
}

void SHOGEvaluator::computeOptFeatures()
{
    int nscales = scaleData->size();
    histogramSizes->resize(nscales);
    histogramData->resize(nscales);
    for (int i = 0; i < nscales; i++)
    {
        const ScaleData& s = getScaleData(i);
        HistData data;
        data.histCols = (s.szi.width - 1) / STEP_SIZE - CELL_SIDE;
        data.histRows = (s.szi.height - 1) / STEP_SIZE - CELL_SIDE;
        data.histSize = data.histCols * data.histRows;
        (*histogramSizes)[i] = data;
        (*histogramData)[i].resize(data.histSize * BINS);
    }
    optfeatures.resize(features->size());
    curScaleIdx = 0;
    HistData& data = (*histogramSizes)[0];
    for (int i = 0; i < optfeatures.size(); i++)
        optfeatures[i].setOffset((*features)[i], data.histSize, data.histCols);

    const ScaleData& s = getScaleData(curScaleIdx);
    gradient->resize((s.szi.width - 1)*(s.szi.height - 1) * BINS);
    integral->resize(s.szi.width * s.szi.height * BINS);
}
