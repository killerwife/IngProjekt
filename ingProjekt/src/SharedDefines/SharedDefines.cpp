#include "SharedDefines.h"
#include <tbb/tbb.h>
#include <opencv2/imgproc.hpp>

void ComputeSHOG(const cv::Mat& image, float* gradient, double* integral, int* histogram, int bins, cv::Size step, cv::Size cell)
{
    char* __restrict imageData = (char*)image.data;
    const int cols = image.cols, rows = image.rows;
    memset(gradient, 0, sizeof(float)*rows*cols * 8);
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

    cv::Mat imageChannels[8] = {
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 0]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 1]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 2]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 3]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 4]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 5]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 6]),
        cv::Mat(rows, cols, CV_32FC1, &gradient[rows*cols * 7]),
    };
    cv::Mat integralChannels[8] = {
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 0]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 1]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 2]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 3]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 4]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 5]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 6]),
        cv::Mat(rows + 1, cols + 1, CV_64F, &integral[(rows + 1)*(cols + 1) * 7]),
    };
    tbb::parallel_for(size_t(0), size_t(8), [&](size_t i) { cv::integral(imageChannels[i], integralChannels[i]); });
    const int sizeRows = cell.height, sizeCols = cell.width;
    const int stepRows = step.height, stepCols = step.width;
    int histSize = (rows / stepRows - sizeRows) * (cols / stepCols - sizeCols);
    tbb::parallel_for(size_t(0), size_t(8), [&](size_t l) {
        for (int i = 0; i < rows - sizeRows; i += stepRows)
        {
            for (int k = 0; k < cols - sizeCols; k += stepCols)
            {
                histogram[l*histSize + i / stepRows*(cols - sizeCols) + k / stepCols] =
                    int(integral[(l*(rows + 1) + i)*(cols + 1) + k] + integral[(l*(rows + 1) + i + sizeRows)*(cols + 1) + k + sizeCols]
                        - integral[(l*(rows + 1) + i)*(cols + 1) + k + sizeCols] - integral[(l*(rows + 1) + i + sizeRows)*(cols + 1) + k]);
            }
        }
    });
}

void outputTimer(std::function<void(void)> func)
{
    auto start = std::chrono::high_resolution_clock::now();

    func();

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << "ns\n";
}