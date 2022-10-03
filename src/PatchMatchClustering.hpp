#ifndef __PATCH_MATCH_CLUSTERING__
#define __PATCH_MATCH_CLUSTERING__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class PatchMatchClustering
{
private:

	struct SegmentCenter
	{
		int num_pixels;
		int sum_xy[2];
		int sum_lab[3];
		float sum_disp;

		SegmentCenter() { num_pixels = sum_xy[0] = sum_xy[1] = sum_lab[0] = sum_lab[1] = sum_lab[2] = 0; sum_disp = 0.f; }
		inline float getPosition(int index) { return (float)sum_xy[index] / num_pixels; }
		inline float getColor(int index) { return (float)sum_lab[index] / num_pixels; }
		inline float getDisparity() { return sum_disp / num_pixels; }
		inline void addPixel(int x, int y, int l, int a, int b, float d)
		{
			num_pixels += 1;
			sum_xy[0] += x;
			sum_xy[1] += y;
			sum_lab[0] += l;
			sum_lab[1] += a;
			sum_lab[2] += b;
			sum_disp += d;
		}
		inline void removePixel(int x, int y, int l, int a, int b, float d)
		{
			num_pixels -= 1;
			sum_xy[0] -= x;
			sum_xy[1] -= y;
			sum_lab[0] -= l;
			sum_lab[1] -= a;
			sum_lab[2] -= b;
			sum_disp -= d;
		}
	};

public:

	PatchMatchClustering();
	~PatchMatchClustering();

	void apply(const cv::Mat3b& image, const cv::Mat1f& disparity_image);
	void enforceLabelConnectivity(int min_size = 25);
	void getLabels(cv::Mat& labels_out);
	int getNumSegments();
	int getSegmentSideLength();

private:

	void initialize(const cv::Mat3b& lab_image, const cv::Mat1f& disparity_image);
	bool isBorder(int x, int y);
	void performClustering(const cv::Mat3b& lab_image, const cv::Mat1f& disparity_image, bool inv = false);
	void iterate(const cv::Mat3b& lab_image, const cv::Mat1f& disparity_image, int max_iteration);

	int w_, h_;
	int num_seg_;
	int segment_size_;
	int segment_side_length_;

	cv::Mat3b lab_image_;
	cv::Mat1f disparity_image_;
	cv::Mat1w segment_label_image_;
	cv::Mat1f cost_matrix_;
	std::vector<SegmentCenter> segment_center_list_;

	float m = 1.f;//30.f;
	float PositionWeight_ = 1.f / (m * m);
	float DisparityWeight_ = 50.f;
	float BoundaryWeight_ = 300.f;
};



#endif
