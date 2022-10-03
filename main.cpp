#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <memory>

#include <libsgm.h>

constexpr int NUM_DISPARITIES = 256;

int main(int argc, char* argv[])
{
	if (argc != 3) {
		std::cout << "usage: " << argv[0] << " [left-image-format] [right-image-format]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	cv::Mat I1, I2;
	I1 = cv::imread(cv::format(argv[1], 0), cv::IMREAD_UNCHANGED);


	sgm::StereoSGM::Parameters param;
	param.LR_max_diff = 1;
	param.path_type = sgm::PathType::SCAN_4PATH;
	param.subpixel = false;
	param.uniqueness = 1.f;

	for (int frame = 0; frame < 200; ++frame)
	{
		std::cout << cv::format("image_2(3)/%06d_10.png\r", frame);

		I1 = cv::imread(cv::format(argv[1], frame), cv::IMREAD_UNCHANGED);
		I2 = cv::imread(cv::format(argv[2], frame), cv::IMREAD_UNCHANGED);
		if (I1.empty() || I2.empty())
			break;

		CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());
		CV_Assert(I1.type() == CV_8UC3 || I1.type() == CV_16UC3);

		if (I1.type() != CV_8UC3)
		{
			I1.convertTo(I1, CV_8UC3);
			I2.convertTo(I2, CV_8UC3);
		}

		cv::Mat G1, G2;
		cv::cvtColor(I1, G1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(I2, G2, cv::COLOR_BGR2GRAY);

		const auto width = I1.cols;
		const auto height = I1.rows;
		cv::Mat1w D1(I1.size()), D2(I1.size());
		sgm::StereoSGM sgm(width, height, NUM_DISPARITIES, 8, 16, sgm::EXECUTE_INOUT_HOST2HOST, param);

		sgm.execute(G1.data, G2.data, D1.data, D2.data);
		D1.setTo(0, D1 > NUM_DISPARITIES);
		D2.setTo(0, D2 > NUM_DISPARITIES);

		cv::Mat D1_color, D2_color;
		D1.convertTo(D1_color, CV_8U, 2);
		D2.convertTo(D2_color, CV_8U, 2);
		cv::applyColorMap(D1_color, D1_color, cv::COLORMAP_JET);
		cv::applyColorMap(D2_color, D2_color, cv::COLORMAP_JET);
		cv::imshow("sgm left", D1_color);
		cv::imshow("sgm right", D2_color);

		const char c = cv::waitKey(0);
		if (c == 27)
			break;
	}

	return 0;
}
