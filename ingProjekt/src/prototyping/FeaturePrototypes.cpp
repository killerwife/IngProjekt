#include "FeaturePrototypes.h"
#include <iostream>
#include "../SharedDefines/SharedDefines.h"
#include "tbb/tbb.h"
#include "tbb/task.h"

FeaturePrototypes::FeaturePrototypes()
{
}


FeaturePrototypes::~FeaturePrototypes()
{
}

void FeaturePrototypes::TestSHOG(cv::Mat& image)
{
    const int cols = image.cols, rows = image.rows;
    float* gradient = new float[rows*cols * 8];
    double* integral = new double[(rows + 1)*(cols + 1) * 8];
    const int sizeRows = 4, sizeCols = 4;
    const int stepRows = 1, stepCols = 1;
    int histSize = (rows / stepRows - (sizeRows) / stepRows + 1) * (cols / stepCols - (sizeCols) / stepCols + 1);
    int* histogram = new int[histSize * 8];
    ComputeSHOG(image,gradient,integral,histogram);
    printf("Gradient Image\n");
    for (int l = 0; l < 8; l++)
    {
        printf("Orientation %d\n", l);
        for (int i = 0; i < rows; i++)
        {
            for (int k = 0; k < cols; k++)
            {
                printf("%3.0f ", gradient[l*rows*cols + i*cols + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    //for (int i = 0; i < 8; i++)
    //    std::cout << imageChannels[i] << std::endl;
    //for (int i = 0; i < 8; i++)
    //    std::cout << integralChannels[i] << std::endl;
    printf("Integral Image\n");
    for (int l = 0; l < 8; l++)
    {
        printf("Orientation %d\n", l);
        for (int i = 0; i < rows + 1; i++)
        {
            for (int k = 0; k < cols + 1; k++)
            {
                printf("%4.0lf ", integral[l*(rows + 1)*(cols + 1) + i*(cols + 1) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    const int histCols = (cols / stepCols - (sizeCols) + 1);
    for (int l = 0; l < 8; l++)
    {
        printf("Orientation %d:\n", l + 1);
        for (int i = 0; i < (rows / stepRows - (sizeRows) / stepRows + 1); i++)
        {
            for (int k = 0; k < histCols; k++)
            {
                printf("%3d ", histogram[l* histSize + i*histCols + k]);
            }
            printf("\n");
        }
    }
    int x = 20, y = 0, ori = 0;
    printf("No offset %d\n", histogram[histCols*x + y + ori * histSize]);
    cv::Point pt(0, 10);
    size_t winStartOffset = pt.x * histCols + pt.y;
    const int* memoryOffset = &histogram[winStartOffset];
    printf("With offset %d\n", memoryOffset[histCols*x + y + ori * histSize]);

    delete gradient;
    delete integral;
    delete histogram;
}

void FeaturePrototypes::ComputeHistFeat(cv::Mat& image)
{

}

void FeaturePrototypes::test()
{
    FeaturePrototypes testing;
    const long rows = 5, cols = 5;
    char* data = new char[rows * cols];
    srand(time(nullptr));
    for (long i = 0; i < rows * cols; ++i)
        data[i] = rand() % 100;

    for (long i = 0; i < rows; i++)
    {
        for (long k = 0; k < cols; k++)
        {
            printf("%d ",data[i*cols + k]);
        }
        printf("\n");
    }

    cv::Mat image(cv::Size(cols, rows), CV_8U, data);
    std::cout << image << std::endl;
    //std::cout << image << std::endl;
    outputTimer([&testing, &image]() {testing.TestSHOG(image); });
    //std::cout << image << std::endl;
    delete data;
}
