#pragma once

#include <string>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

class LaneDetector {

private:
	double img_size;
	double img_center;
	bool left_flag = false;
	bool right_flag = false;
	Point right_b;
	double right_m;
	Point left_b;
	double left_m;

public:
	Mat cannyImage(Mat inputImage);
	Mat regionOfInterest(Mat img_canny);
	std::vector<Vec4i> houghLines(Mat img_roi);
	std::vector<std::vector<Vec4i>>lineSeparation(std::vector<Vec4i> lines, Mat img_edges);
	std::vector<Point> regression(std::vector < std:: vector<Vec4i >> left_right_lines, Mat inputImage);
	std::string predictTurn();
	int plotLine(Mat inputImage, std::vector<Point> lane, std::string turn);
};