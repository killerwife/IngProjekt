#include "SHOGfeatures.h"
#include "opencv2/core.hpp"
#include <tbb/tbb.h>
#include <opencv2/imgproc.hpp>

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
    winStartOffset = pt.x * (*histogramSizes)[curScaleIdx].histCols + pt.y;
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
    char* __restrict imageData = (char*)imageMat.data;
    const int cols = imageMat.cols, rows = imageMat.rows;
    float* gradient = this->gradient->data();
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
    //printf("Gradient:\n");
    //for (int i = 0; i < 8; i++)
    //{
    //    printf("Orientation %d:\n",i);
    //    for (int k = 0; k < rows; k++)
    //    {
    //        printf("Row num %2d:", k);
    //        for (int l = 0; l < cols; l++)
    //        {
    //            printf("%4.0f ", gradient[rows*cols*i + k*cols + l]);
    //        }
    //        printf("\n");
    //    }
    //}
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
    double* integral = this->integral->data();
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
    //printf("Integral Image:\n");
    //for (int i = 0; i < 8; i++)
    //{
    //    printf("Orientation %d:\n", i);
    //    for (int k = 0; k < rows + 1; k++)
    //    {
    //        printf("Row num %2d:", k);
    //        for (int l = 0; l < cols + 1; l++)
    //        {
    //            printf("%4.0lf ", integral[(rows + 1)*(cols + 1)*i + k*(cols + 1) + l]);
    //        }
    //        printf("\n");
    //    }
    //}
    const int sizeRows = CELL_SIDE, sizeCols = CELL_SIDE;
    const int stepRows = STEP_SIZE, stepCols = STEP_SIZE;
    HistData& data = (*histogramSizes)[i];
    int* memory = (*histogramData)[i].data();
    tbb::parallel_for(size_t(0), size_t(7), [&](size_t l) {
        for (int i = 0; i < rows - sizeRows; i += stepRows)
        {
            for (int k = 0; k < cols - sizeCols; k += stepCols)
            {
                memory[l*data.histSize + i / stepRows*(cols - sizeCols) + k / stepCols] =
                    int(integral[(l*(rows + 1) + i)*(cols + 1) + k] + integral[(l*(rows + 1) + i + sizeRows)*(cols + 1) + k + sizeCols]
                        - integral[(l*(rows + 1) + i)*(cols + 1) + k + sizeCols] - integral[(l*(rows + 1) + i + sizeRows)*(cols + 1) + k]);
            }
        }
    });
    //printf("Histogram:\n");
    //for (int i = 0; i < 8; i++)
    //{
    //    printf("Orientation %d:\n", i);
    //    for (int k = 0; k < data.histRows; k++)
    //    {
    //        printf("Row num %2d:", k);
    //        for (int l = 0; l < data.histCols; l++)
    //        {
    //            printf("%4d ", memory[data.histSize*i + k*data.histCols + l]);
    //        }
    //        printf("\n");
    //    }
    //}
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
