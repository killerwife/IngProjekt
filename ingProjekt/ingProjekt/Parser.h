#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  
class Parser
{
public:
    Parser();
    ~Parser();
    void Parser::parseNegatives();
    void Parser::parsePositives(); 
    void Parser::toMat(cv::Mat** output, cv::Mat** responses, std::string location, int count, cv::Mat response);
};

