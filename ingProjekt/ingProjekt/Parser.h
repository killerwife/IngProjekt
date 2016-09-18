#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  
class Parser
{
public:
    Parser();
    ~Parser();
    void Parser::parseNegatives(); // Parses negative pictures from a hardcoded folder 
    void Parser::parsePositives(); // Parses positive pictures from a hardcoded folder
	/*
	Function, which opens a folder, and picks the first n pictures, up to count. It inserts one line into MATs output
	and responses per picture loaded. Response is generally 0 or 1, based on whether given folder contains positive or negative pictures.
	WARNING: Windows only solution
	*/
    void Parser::toMat(cv::Mat** output, cv::Mat** responses, std::string location, int count, cv::Mat response);
};

