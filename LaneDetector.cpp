#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "LaneDetector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


Mat LaneDetector::cannyImage(Mat inputImage)
{
	Mat gray, blur, canny;
	cvtColor(inputImage, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(5, 5), 0);
	Canny(blur, canny, 50, 150);
	return canny;
}


Mat LaneDetector::regionOfInterest(Mat img_canny)
{

	Mat output;
	Mat mask = Mat::zeros(img_canny.size(), img_canny.type());

	Point pts[4] = {

		Point(100,1000), //LEFT-BOTTOM
		Point(850,250), //LEFT UPPER
		Point(850, 250), //RIGHT UPPER
		Point(1850,1000) //RIGHT-BOTTOM
	};

	fillConvexPoly(mask, pts, 4, Scalar(255, 0, 0));
	bitwise_and(img_canny, mask, output);
	return output;
}


std::vector<Vec4i> LaneDetector::houghLines(Mat img_roi)
{
	std::vector<Vec4i> line;
	HoughLinesP(img_roi, line, 1, CV_PI / 180, 20, 20, 30);

	return line;
}


std::vector<std::vector<Vec4i>> LaneDetector::lineSeparation(std::vector<Vec4i>lines, Mat img_edges)
{
	std::vector<std::vector<Vec4i>> output(2);
	size_t j = 0;
	Point ini;
	Point fini;
	double slope_thresh = 0.3;
	std::vector<double>slopes;
	std::vector<Vec4i>selected_lines;
	std::vector<Vec4i> right_lines, left_lines;

	for (auto i : lines) {
		ini = Point(i[0], i[1]);
		fini = Point(i[2], i[3]);

		double slope = (
			static_cast<double>(fini.y) - static_cast<double>(ini.y)) 
			/ 
			(static_cast<double>(fini.x) - static_cast<double>
			(ini.x));
		if (std::abs(slope) > slope_thresh)
		{
			slopes.push_back(slope);
			selected_lines.push_back(i);
		}
	}
	img_center = static_cast<double>((img_edges.cols / 2));
	while (j < selected_lines.size())
	{
		ini = Point(selected_lines[j][0], selected_lines[j][1]);
		fini = Point(selected_lines[j][2], selected_lines[j][3]);
		if (slopes[j] > 0 && fini.x > img_center && ini.x > img_center)
		{
			right_lines.push_back(selected_lines[j]);
			right_flag = true;
		}
		else if (slopes[j] < 0 && fini.x < img_center && ini.x < img_center)
		{
			left_lines.push_back(selected_lines[j]);
			left_flag = true;
		}
		j++;
	}
	output[0] = right_lines;
	output[1] = left_lines;
	return output;

}


std::vector<cv::Point> LaneDetector::regression(std::vector<std::vector<Vec4i> > left_right_lines, Mat inputImage) {
	std::vector<cv::Point> output(4);
	Point ini;
	Point fini;
	Point ini2;
	Point fini2;
	Vec4d right_line;
	Vec4d left_line;
	std::vector<cv::Point> right_pts;
	std::vector<cv::Point> left_pts;

	// If right lines are being detected, fit a line using all the init and final points of the lines
	if (right_flag == true) {
		for (auto i : left_right_lines[0]) {
			ini = cv::Point(i[0], i[1]);
			fini = cv::Point(i[2], i[3]);

			right_pts.push_back(ini);
			right_pts.push_back(fini);
		}

		if (right_pts.size() > 0) {
			// The right line is formed here
			cv::fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01);
			right_m = right_line[1] / right_line[0];
			right_b = cv::Point(right_line[2], right_line[3]);
		}
	}

	// If left lines are being detected, fit a line using all the init and final points of the lines
	if (left_flag == true) {
		for (auto j : left_right_lines[1]) {
			ini2 = cv::Point(j[0], j[1]);
			fini2 = cv::Point(j[2], j[3]);

			left_pts.push_back(ini2);
			left_pts.push_back(fini2);
		}

		if (left_pts.size() > 0) {
			// The left line is formed here
			cv::fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);
			left_m = left_line[1] / left_line[0];
			left_b = cv::Point(left_line[2], left_line[3]);
		}
	}

	// One the slope and offset points have been obtained, apply the line equation to obtain the line points
	int ini_y = inputImage.rows;
	int fin_y = 470;

	double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
	double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

	double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
	double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

	output[0] = cv::Point(right_ini_x, ini_y);
	output[1] = cv::Point(right_fin_x, fin_y);
	output[2] = cv::Point(left_ini_x, ini_y);
	output[3] = cv::Point(left_fin_x, fin_y);

	return output;
}




std::string LaneDetector::predictTurn() {
	std::string output;
	double vanish_x;
	double thr_vp = 10;

	// The vanishing point is the point where both lane boundary lines intersect
	vanish_x = static_cast<double>(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

	// The vanishing points location determines where is the road turning
	if (vanish_x < (img_center - thr_vp))
		output = "Left Turn";
	else if (vanish_x > (img_center + thr_vp))
		output = "Right Turn";
	else if (vanish_x >= (img_center - thr_vp) && vanish_x <= (img_center + thr_vp))
		output = "Straight";

	return output;
}



int LaneDetector::plotLine(Mat inputImage, std::vector<Point> lane, std::string turn)
{

	std::vector<Point> poly_points;
	Mat output;

	inputImage.copyTo(output);

	poly_points.push_back(lane[2]);
	poly_points.push_back(lane[0]);
	poly_points.push_back(lane[1]);
	poly_points.push_back(lane[3]);
	
	fillConvexPoly(output, poly_points, Scalar(0, 0, 255), LINE_AA, 0);
	addWeighted(output, 0.3, inputImage, 1.0 - 0.3, 0, inputImage);

	line(inputImage, lane[0], lane[1], Scalar(0, 255, 255), 5, LINE_AA);
	line(inputImage, lane[2], lane[3], Scalar(0, 255, 255), 5, LINE_AA);

	putText(inputImage, turn, Point(50, 90), FONT_HERSHEY_COMPLEX_SMALL, 3, Scalar(0, 255, 0), 1, LINE_AA);


	imshow("test", inputImage);

	return 0;
}

int main()
{

	VideoCapture cap("MyVideo.mp4");

	LaneDetector lanedetector;
	Mat frame, img_denoise, img_edges, img_mask, img_lines;

	std::vector<Vec4i> lines;
	std::vector<std::vector<Vec4i>> left_right_lines;
	std::vector<Point> lane;
	std::string turn;
	int flag_plot = -1;
	int i = 0;

	while (i < 540)
	{
		if (!cap.read(frame))
		{
			break;
		}
		img_edges = lanedetector.cannyImage(frame);
		img_mask = lanedetector.regionOfInterest(img_edges);
		lines = lanedetector.houghLines(img_mask);

		if (!lines.empty())
		{
			left_right_lines = lanedetector.lineSeparation(lines, img_edges);
			lane = lanedetector.regression(left_right_lines, frame);
			turn = lanedetector.predictTurn();
			flag_plot = lanedetector.plotLine(frame, lane, turn);
			i += 1;
			waitKey(25);
		}
		else {
			flag_plot = -1;
		}
	}
	return flag_plot;

}

