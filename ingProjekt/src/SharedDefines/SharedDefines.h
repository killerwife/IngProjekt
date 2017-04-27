#pragma once
#include <chrono>
#include <iostream>
#include <functional>
#include <opencv2/core.hpp>

void outputTimer(std::function<void(void)> func);
void ComputeSHOG(const cv::Mat& image, float* gradient, double* integral, int* histogram, int bins = 8, cv::Size step = cv::Size(1, 1), cv::Size cell = cv::Size(4, 4));