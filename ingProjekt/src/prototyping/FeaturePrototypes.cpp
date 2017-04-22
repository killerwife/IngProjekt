#include "FeaturePrototypes.h"
#include <iostream>
#include "SharedDefines\SharedDefines.h"
#include "tbb/tbb.h"
#include "tbb/task.h"

FeaturePrototypes::FeaturePrototypes()
{
}


FeaturePrototypes::~FeaturePrototypes()
{
}

void FeaturePrototypes::ComputeSHog(cv::Mat& image)
{
    int* __restrict imageData = (int*)image.data;
    const int cols = image.cols, rows = image.rows;
    float* images = new float[rows*cols * 8];
    memset(images, 0, sizeof(float)*rows*cols * 8);
#pragma loop(hint_parallel(8))
#pragma loop(ivdep)
    for (long i = 1; i < rows - 1; i++)
    {
        for (long k = 1; k < cols - 1; k++)
        {
            int dx = -imageData[i*cols + k - 1] + imageData[i*cols + k + 1];
            int dy = -imageData[(i - 1)*cols + k] + imageData[(i + 1)*cols + k];
            int absol1 = std::abs(dy);
            int absol2 = std::abs(dx);
            int orientation = (dy < 0) * 4 + (dx < 0) * 2 + (absol1 > absol2) * 1;
            images[rows*cols*orientation + i*cols + k] = float(absol1 + absol2);
        }
    }
    printf("Gradient Image\n");
    for (int l = 0; l < 8; l++)
    {
        printf("Orientation %d\n", l);
        for (int i = 0; i < rows; i++)
        {
            for (int k = 0; k < cols; k++)
            {
                printf("%.2f ", images[l*rows*cols + i*cols + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    cv::Mat imageChannels[8] = {
        cv::Mat(rows, cols, CV_32FC1, &images[rows*cols * 0]),
        cv::Mat(rows, cols, CV_32FC1, &images[rows*cols * 1]),
        cv::Mat(rows, cols, CV_32FC1, &images[rows*cols * 2]),
        cv::Mat(rows, cols, CV_32FC1, &images[rows*cols * 3]),
        cv::Mat(rows, cols, CV_32FC1, &images[rows*cols * 4]),
        cv::Mat(rows, cols, CV_32FC1, &images[rows*cols * 5]),
        cv::Mat(rows, cols, CV_32FC1, &images[rows*cols * 6]),
        cv::Mat(rows, cols, CV_32FC1, &images[rows*cols * 7]),
    };
    images[2] = 5;
    for (int i = 0; i < 8; i++)
        std::cout << imageChannels[i] << std::endl;
    double* integral = new double[(rows + 1)*(cols + 1) * 8];
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
    tbb::parallel_for(size_t(0), size_t(7), [&](size_t i) { cv::integral(imageChannels[i], integralChannels[i]); });
    for (int i = 0; i < 8; i++)
        std::cout << integralChannels[i] << std::endl;
    printf("Integral Image\n");
    for (int l = 0; l < 8; l++)
    {
        printf("Orientation %d\n", l);
        for (int i = 0; i < rows + 1; i++)
        {
            for (int k = 0; k < cols + 1; k++)
            {
                printf("%.2lf ", integral[l*(rows + 1)*(cols + 1) + i*(cols+1) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    //for(int i =0;i<8;i++)
    //    cv::integral();

    const int sizeRows = 4, sizeCols = 4;
    const int stepRows = 1, stepCols = 1;
    int histSize = (rows / stepRows - (sizeRows - 1) / stepRows) * (cols / stepCols - (sizeCols - 1) / stepCols);
    long long* memory = new long long[histSize * 8];
    outputTimer([&]() {
        tbb::parallel_for(size_t(0), size_t(7), [&](size_t l) {
            for (int i = 0; i < rows - sizeRows; i += stepRows)
            {
                for (int k = 0; k < cols - sizeCols; k += stepCols)
                {
                    memory[l*histSize + i / stepRows*(cols - sizeCols) + k / stepCols] =
                        long long(integral[(l*(rows+1) + i)*(cols+1) + k] + integral[(l*(rows + 1) + i + sizeRows)*(cols + 1) + k + sizeCols]
                            - integral[(l*(rows + 1) + i)*(cols + 1) + k + sizeCols] - integral[(l*(rows + 1) + i + sizeRows)*(cols + 1) + k]);
                }
            }
        });
    });
    const int histCols = (cols / stepCols - (sizeCols - 1));
    for (int l = 0; l < 8; l++)
    {
        printf("Orientation %d:\n", l + 1);
        for (int i = 0; i < (rows / stepRows - (sizeRows - 1) / stepRows); i++)
        {
            for (int k = 0; k < histCols; k++)
            {
                printf("%lld ", memory[l* histSize + i*histCols + k]);
            }
            printf("\n");
        }
    }
    delete memory;
    delete images;
    delete integral;
}

void FeaturePrototypes::ComputeHistFeat(cv::Mat& image)
{

}

void FeaturePrototypes::test()
{
    FeaturePrototypes testing;
    const long rows = 20, cols = 50;
    int* data = new int[rows * cols];
    for (long i = 0; i < rows * cols; ++i)
        data[i] = i;

    cv::Mat image(cv::Size(cols, rows), CV_32S, data);
    //std::cout << image << std::endl;
    outputTimer([&testing, &image]() {testing.ComputeSHog(image); });
    //std::cout << image << std::endl;
    delete data;
}
