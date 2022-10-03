#include "PatchMatchClustering.hpp"
#include <stack>

#include <iostream>
#include <opencv2/highgui.hpp>

const int gDx[8] = { +1, +0, -1, +0, +1, -1, +1, -1 };
const int gDy[8] = { +0, +1, +0, -1, +1, +1, -1, -1 };

static int DivUp(int a, int b) { return (a + b - 1) / b; }
template<typename T> static T Square(T n) { return n * n; }
static cv::Mat drawLabelBorder(const cv::Mat1w& label_image)
{
	cv::Mat3b border_image(label_image.size(), cv::Vec3b(0u, 0u, 0u));
	border_image.forEach([&](cv::Vec3b& val, const int* p)
	{
		const int x = p[1], y = p[0];
		if (x + 1 < border_image.cols && label_image.ptr<ushort>(y)[x] != label_image.ptr<ushort>(y)[x + 1])
			val[2] = 255u;
		if (y + 1 < border_image.rows && label_image.ptr<ushort>(y)[x] != label_image.ptr<ushort>(y + 1)[x])
			val[2] = 255u;
	});
	return border_image;
}
static cv::Mat drawLabelBorder(const cv::Mat3b& image, const cv::Mat1w& label_image)
{
	cv::Mat3b border_image = drawLabelBorder(label_image);
	cv::Mat3b ret;
	image.copyTo(ret);
	ret.forEach([&](cv::Vec3b& val, const int* p)
	{
		if (border_image.ptr<cv::Vec3b>(p[0])[p[1]] == cv::Vec3b(0u, 0u, 255u))
			val = cv::Vec3b(0u, 0u, 255u);
	});
	return ret;
}

PatchMatchClustering::PatchMatchClustering()
{}

PatchMatchClustering::~PatchMatchClustering()
{}

void PatchMatchClustering::initialize(const cv::Mat3b& lab_image, const cv::Mat1f& disparity_image)
{
	segment_label_image_.create(h_, w_);
	cost_matrix_.create(h_, w_);

	segment_size_ = DivUp(w_ * h_, 1000);
	segment_side_length_ = (int)std::sqrt((float)segment_size_);
	num_seg_ = DivUp(w_, segment_side_length_) * DivUp(h_, segment_side_length_);
	segment_center_list_.resize(num_seg_);

	const int segment_step = DivUp(w_, segment_side_length_);

	for (int y = 0; y < h_; ++y)
	{
		for (int x = 0; x < w_; ++x)
		{
			const int segment_row = y / segment_side_length_;
			const int segment_col = x / segment_side_length_;
			const int segment_index = segment_row * segment_step + segment_col;
			segment_label_image_.ptr<ushort>(y)[x] = segment_index;

			SegmentCenter& center = segment_center_list_[segment_index];
			center.num_pixels += 1;
			center.sum_xy[0] += x;
			center.sum_xy[1] += y;
			center.sum_lab[0] += lab_image.ptr<cv::Vec3b>(y)[x][0];
			center.sum_lab[1] += lab_image.ptr<cv::Vec3b>(y)[x][1];
			center.sum_lab[2] += lab_image.ptr<cv::Vec3b>(y)[x][2];
			center.sum_disp += disparity_image.ptr<float>(y)[x];
		}
	}

	for (int y = 0; y < h_; ++y)
	{
		for (int x = 0; x < w_; ++x)
		{
			const int segment_index = segment_label_image_.ptr<ushort>(y)[x];
			SegmentCenter& center = segment_center_list_[segment_index];

			const float color_cost = Square(lab_image.ptr<cv::Vec3b>(y)[x][0] - center.getColor(0))
								   + Square(lab_image.ptr<cv::Vec3b>(y)[x][1] - center.getColor(1))
								   + Square(lab_image.ptr<cv::Vec3b>(y)[x][2] - center.getColor(2));
			const float position_cost = Square(x - center.getPosition(0))
									  + Square(y - center.getPosition(1));
			const float disparity_cost = Square(disparity_image.ptr<float>(y)[x] - center.getDisparity());

			cost_matrix_.ptr<float>(y)[x] = color_cost + PositionWeight_ * position_cost + DisparityWeight_ * disparity_cost;
		}
	}
}

inline bool PatchMatchClustering::isBorder(int x, int y)
{
	for (int i = 0; i < 4; ++i)
		if (x + gDx[i] >= 0 && x + gDx[i] < w_ && y + gDy[i] >= 0 && y + gDy[i] < h_ &&
			segment_label_image_.ptr<ushort>(y)[x] != segment_label_image_.ptr<ushort>(y + gDy[i])[x + gDx[i]])
			return true;
	return false;
}

void PatchMatchClustering::performClustering(const cv::Mat3b& lab_image, const cv::Mat1f& disparity_image, bool inv)
{
	const int sx = inv ? w_ - 1 : 0;
	const int sy = inv ? h_ - 1 : 0;
	const int ex = inv ? -1 : w_;
	const int ey = inv ? -1 : h_;
	const int inc = inv ? -1 : 1;

	for (int y = sy; y != ey; y += inc)
	{
		for (int x = sx; x != ex ; x += inc)
		{
			for (int i = 0; i < 4; ++i)
			{
				const int process_index = segment_label_image_.ptr<ushort>(y)[x];
				const int nx = x + gDx[i];
				const int ny = y + gDy[i];
				if (nx < 0 || nx >= w_ || ny < 0 || ny >= h_) continue;

				const int neighbor_index = segment_label_image_.ptr<ushort>(ny)[nx];
				if (process_index == neighbor_index) continue;

				SegmentCenter& center = segment_center_list_[neighbor_index];

				const float color_cost = Square(lab_image.ptr<cv::Vec3b>(y)[x][0] - center.getColor(0))
									   + Square(lab_image.ptr<cv::Vec3b>(y)[x][1] - center.getColor(1))
									   + Square(lab_image.ptr<cv::Vec3b>(y)[x][2] - center.getColor(2));
				const float position_cost = Square(x - center.getPosition(0))
										  + Square(y - center.getPosition(1));
				const float disparity_cost = Square(disparity_image.ptr<float>(y)[x] - center.getDisparity());

				int boundary_cost = 8;
				for (int n = 0; n < 8; ++n)
					if (x + gDx[n] >= 0 && x + gDx[n] < w_ && y + gDy[n] >= 0 && y + gDy[n] < h_
					&& segment_label_image_.ptr<ushort>(y + gDy[n])[x + gDx[n]] == neighbor_index)
						--boundary_cost;
				//boundary_cost = std::abs(boundary_cost - 4);

				const float total_cost = color_cost
									   + PositionWeight_ * position_cost
									   + DisparityWeight_ * disparity_cost
									   + BoundaryWeight_ * boundary_cost;

				if (cost_matrix_.ptr<float>(y)[x] > total_cost)
				{
					cost_matrix_.ptr<float>(y)[x] = total_cost;
					segment_label_image_.ptr<ushort>(y)[x] = neighbor_index;

					const int l = lab_image.ptr<cv::Vec3b>(y)[x][0];
					const int a = lab_image.ptr<cv::Vec3b>(y)[x][1];
					const int b = lab_image.ptr<cv::Vec3b>(y)[x][2];
					const float d =  disparity_image.ptr<float>(y)[x];

					segment_center_list_[process_index].removePixel(x, y, l, a, b, d);
					segment_center_list_[neighbor_index].addPixel(x, y, l, a, b, d);
				}
			}
		}
	}
}

void PatchMatchClustering::iterate(const cv::Mat3b& lab_image, const cv::Mat1f& disparity_image, int max_iteration)
{
	for (int itr = 0; itr < max_iteration; ++itr)
	{
		performClustering(lab_image, disparity_image, itr % 2 != 0);
	}
}

void PatchMatchClustering::apply(const cv::Mat3b& image, const cv::Mat1f& disparity_image)
{
	CV_Assert(image.size() == disparity_image.size());

	h_ = image.rows;
	w_ = image.cols;
	cv::cvtColor(image, lab_image_, cv::COLOR_BGR2Lab);
	disparity_image.copyTo(disparity_image_);

	initialize(lab_image_, disparity_image_);
	iterate(lab_image_, disparity_image_, 10);

	// cv::imshow("test", drawLabelBorder(segment_label_image_));
}

void PatchMatchClustering::enforceLabelConnectivity(int min_size)
{
	int changed_count;
	do
	{
		changed_count = 0;
		for (int y = 0; y < h_; ++y)
		{
			for (int x = 0; x < w_; ++x)
			{
				const int segment_idx = segment_label_image_.ptr<ushort>(y)[x];
				if (segment_center_list_[segment_idx].num_pixels < min_size)
				{
					int new_segment_idx = -1;
					float min_color_diff = 9999.f;
					for (int i = 0; i != 8; ++i)
					{
						const int nx = x + gDx[i];
						const int ny = y + gDy[i];
						if (nx < 0 || nx >= w_ || ny < 0 || ny >= h_) continue;

						const int n_segment_idx = segment_label_image_.ptr<ushort>(ny)[nx];

						if (segment_idx != n_segment_idx && segment_center_list_[n_segment_idx].num_pixels > min_size)
						{
							SegmentCenter& segment1 = segment_center_list_[segment_idx];
							SegmentCenter& segment2 = segment_center_list_[n_segment_idx];

							float color_diff = std::abs(segment1.getColor(0) - segment2.getColor(0))
											 + std::abs(segment1.getColor(1) - segment2.getColor(1))
											 + std::abs(segment1.getColor(2) - segment2.getColor(2));

							if (min_color_diff > color_diff)
							{
								min_color_diff = color_diff;
								new_segment_idx = n_segment_idx;
							}
						}
					}

					if (new_segment_idx >= 0)
					{
						segment_label_image_.ptr<ushort>(y)[x] = new_segment_idx;
						changed_count += 1;

						const int l = lab_image_.ptr<cv::Vec3b>(y)[x][0];
						const int a = lab_image_.ptr<cv::Vec3b>(y)[x][1];
						const int b = lab_image_.ptr<cv::Vec3b>(y)[x][2];
						const float d = disparity_image_.ptr<float>(y)[x];
						segment_center_list_[new_segment_idx].addPixel(x, y, l, a, b, d);
						segment_center_list_[segment_idx].removePixel(x, y, l, a, b, d);
					}
				}
			}
		}
	} while (changed_count > 0);
}

void PatchMatchClustering::getLabels(cv::Mat& labels_out) { segment_label_image_.copyTo(labels_out); }
int PatchMatchClustering::getNumSegments() { return num_seg_; }
int PatchMatchClustering::getSegmentSideLength() { return segment_side_length_; }
