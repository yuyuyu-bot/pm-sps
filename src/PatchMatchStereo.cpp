#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <random>
#include <ctime>
#include <iostream>

#include "PatchMatchStereo.hpp"
#include "defines.hpp"

const int dx[4] = { +1, +0, -1, +0 };
const int dy[4] = { +0, +1, +0, -1 };

void makeSegmentBoundaryImage(const cv::Mat3b& inputImage,
							  const cv::Mat1w& segmentImage,
							  std::vector<std::vector<int>>& boundaryLabels,
							  cv::Mat3b& segmentBoundaryImage)
{
	int width = inputImage.cols;
	int height = inputImage.rows;
	int boundaryTotal = static_cast<int>(boundaryLabels.size());

	inputImage.copyTo(segmentBoundaryImage);

	int boundaryWidth = 2;
	for (int y = 0; y < height - 1; ++y) {
		for (int x = 0; x < width - 1; ++x) {
			int pixelLabelIndex = segmentImage.ptr<ushort>(y)[x];

			if (segmentImage.ptr<ushort>(y)[x + 1] != pixelLabelIndex) {
				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (x - w >= 0) segmentBoundaryImage.ptr<cv::Vec3b>(y)[x - w] = { 128u, 128u, 128u };
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (x + w < width) segmentBoundaryImage.ptr<cv::Vec3b>(y)[x + w] = { 128u, 128u, 128u };
				}
			}
			if (segmentImage.ptr<ushort>(y + 1)[x] != pixelLabelIndex) {
				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (y - w >= 0) segmentBoundaryImage.ptr<cv::Vec3b>(y - w)[x] = { 128u, 128u, 128u };
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (y + w < height) segmentBoundaryImage.ptr<cv::Vec3b>(y + w)[x] = { 128u, 128u, 128u };
				}
			}
		}
	}

	boundaryWidth = 7;
	for (int y = 0; y < height - 1; ++y) {
		for (int x = 0; x < width - 1; ++x) {
			int pixelLabelIndex = segmentImage.ptr<ushort>(y)[x];

			if (segmentImage.ptr<ushort>(y)[x + 1] != pixelLabelIndex) {
				cv::Vec3b negativeSideColor, positiveSideColor;
				int pixelBoundaryIndex = -1;
				for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
					if ((boundaryLabels[boundaryIndex][0] == pixelLabelIndex && boundaryLabels[boundaryIndex][1] == segmentImage.ptr<ushort>(y)[x + 1])
						|| (boundaryLabels[boundaryIndex][0] == segmentImage.ptr<ushort>(y)[x + 1] && boundaryLabels[boundaryIndex][1] == pixelLabelIndex))
					{
						pixelBoundaryIndex = boundaryIndex;
						break;
					}
				}

				if (boundaryLabels[pixelBoundaryIndex][2] == 3)
					continue;
				else if (boundaryLabels[pixelBoundaryIndex][2] == 2) {
					negativeSideColor[2] = 0;  negativeSideColor[1] = 225;  negativeSideColor[0] = 0;
					positiveSideColor[2] = 0;  positiveSideColor[1] = 225;  positiveSideColor[0] = 0;
				} else if (pixelLabelIndex == boundaryLabels[pixelBoundaryIndex][boundaryLabels[pixelBoundaryIndex][2]]) {
					negativeSideColor[2] = 225;  negativeSideColor[1] = 0;  negativeSideColor[0] = 0;
					positiveSideColor[2] = 0;  positiveSideColor[1] = 0;  positiveSideColor[0] = 225;
				} else {
					negativeSideColor[2] = 0;  negativeSideColor[1] = 0;  negativeSideColor[0] = 225;
					positiveSideColor[2] = 225;  positiveSideColor[1] = 0;  positiveSideColor[0] = 0;
				}

				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (x - w >= 0) segmentBoundaryImage.ptr<cv::Vec3b>(y)[x - w] = negativeSideColor;
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (x + w < width) segmentBoundaryImage.ptr<cv::Vec3b>(y)[x + w] = positiveSideColor;
				}
			}
			if (segmentImage.ptr<ushort>(y + 1)[x] != pixelLabelIndex) {
				cv::Vec3b negativeSideColor, positiveSideColor;
				int pixelBoundaryIndex = -1;
				for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
					if ((boundaryLabels[boundaryIndex][0] == pixelLabelIndex && boundaryLabels[boundaryIndex][1] == segmentImage.ptr<ushort>(y + 1)[x])
						|| (boundaryLabels[boundaryIndex][0] == segmentImage.ptr<ushort>(y + 1)[x] && boundaryLabels[boundaryIndex][1] == pixelLabelIndex))
					{
						pixelBoundaryIndex = boundaryIndex;
						break;
					}
				}
				if (boundaryLabels[pixelBoundaryIndex][2] == 3) continue;
				else if (boundaryLabels[pixelBoundaryIndex][2] == 2) {
					negativeSideColor[2] = 0;  negativeSideColor[1] = 225;  negativeSideColor[0] = 0;
					positiveSideColor[2] = 0;  positiveSideColor[1] = 225;  positiveSideColor[0] = 0;
				} else if (pixelLabelIndex == boundaryLabels[pixelBoundaryIndex][boundaryLabels[pixelBoundaryIndex][2]]) {
					negativeSideColor[2] = 225;  negativeSideColor[1] = 0;  negativeSideColor[0] = 0;
					positiveSideColor[2] = 0;  positiveSideColor[1] = 0;  positiveSideColor[0] = 225;
				} else {
					negativeSideColor[2] = 0;  negativeSideColor[1] = 0;  negativeSideColor[0] = 225;
					positiveSideColor[2] = 225;  positiveSideColor[1] = 0;  positiveSideColor[0] = 0;
				}

				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (y - w >= 0) segmentBoundaryImage.ptr<cv::Vec3b>(y - w)[x] = negativeSideColor;
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (y+ w < height) segmentBoundaryImage.ptr<cv::Vec3b>(y + w)[x] = positiveSideColor;
				}
			}
		}
	}
}

static inline bool inBound(int x, int y, int width, int height)
{
	return x >= 0 && y >= 0 && x < width && y < height;
}

static inline int getRandom(int min, int max)
{
	return (std::rand() % (max - min + 1)) + min;
}

static inline float getRandomf(float min, float max)
{
	return std::rand() * (max - min) / RAND_MAX + min;
}

template <int VIEW = 0>
static void symmetricCensus9x7(const cv::Mat& src, cv::Mat& dst)
{
	CV_Assert(dst.elemSize() == 4);

	memset(dst.data, 0, dst.rows * dst.cols * sizeof(uint32_t));

	const int RADIUS_U = 9 / 2;
	const int RADIUS_V = 7 / 2;

	int v;
// #pragma omp parallel for
	for (v = RADIUS_V; v < src.rows - RADIUS_V; v++)
	{
		uint32_t* dstptr = dst.ptr<uint32_t>(v);
		for (int u = RADIUS_U; u < src.cols - RADIUS_U; u++)
		{
			uint32_t c = 0;
			for (int dv = -RADIUS_V; dv <= -1; dv++)
			{
				for (int du = -RADIUS_U; du <= RADIUS_U; du++)
				{
					const int v1 = v + dv;
					const int v2 = v - dv;
					const int u1 = u + du;
					const int u2 = u - du;
					c <<= 1;
					c += src.ptr(v1)[u1] <= src.ptr(v2)[u2] ? 0 : 1;
				}
			}
			{
				int dv = 0;
				for (int du = -RADIUS_U; du <= -1; du++)
				{
					const int v1 = v + dv;
					const int v2 = v - dv;
					const int u1 = u + du;
					const int u2 = u - du;
					c <<= 1;
					c += src.ptr(v1)[u1] <= src.ptr(v2)[u2] ? 0 : 1;
				}
			}
			if (VIEW == 0)
				dstptr[u] = c;
			else
				dstptr[src.cols - 1 - u] = c;
		}
	}
}

static inline int L1_norm(const cv::Vec3b& a)
{
	return std::abs(a[0]) + std::abs(a[1]) + std::abs(a[2]);
}

static inline int HammingDistance(uint32_t c1, uint32_t c2)
{
	return static_cast<int>(popcnt32(c1 ^ c2));
}

static inline int computeRequiredSamplingNum(int num_draw, int num_inlier, int num_data, int num_sampling, float confidence_level)
{
	float ep = 1.f - (float)num_inlier / num_data;
	if (ep == 1.f) ep = .5f;
	int new_num_sampling = (int)(std::log(1.f - confidence_level) / std::log(1.f - std::pow(1.f - ep, num_draw)) + .5f);
	return std::min(num_sampling, new_num_sampling);
}

static void solvePlaneEquations(const float x1, const float y1, const float z1, const float d1,
								const float x2, const float y2, const float z2, const float d2,
								const float x3, const float y3, const float z3, const float d3,
								std::vector<float>& planeParameter)
{
	const float epsilonValue = 1e-10f;

	planeParameter.resize(3);

	float denominatorA = (x1 * z2 - x2 * z1) * (y2 * z3 - y3 * z2) - (x2 * z3 - x3 * z2) * (y1 * z2 - y2 * z1);
	if (denominatorA < epsilonValue)
	{
		planeParameter[0] = 0.0;
		planeParameter[1] = 0.0;
		planeParameter[2] = -1.0;
		return;
	}

	planeParameter[0] = ((z2 * d1 - z1 * d2) * (y2 * z3 - y3 * z2) - (z3 * d2 - z2 * d3) * (y1 * z2 - y2 * z1)) / denominatorA;

	float denominatorB = y1 * z2 - y2 * z1;
	if (denominatorB > epsilonValue)
		planeParameter[1] = (z2 * d1 - z1 * d2 - planeParameter[0] * (x1 * z2 - x2 * z1)) / denominatorB;
	else
	{
		denominatorB = y2 * z3 - y3 * z2;
		planeParameter[1] = (z3 * d2 - z2 * d3 - planeParameter[0] * (x2 * z3 - x3 * z2)) / denominatorB;
	}
	if (z1 > epsilonValue)
		planeParameter[2] = (d1 - planeParameter[0] * x1 - planeParameter[1] * y1) / z1;
	else if (z2 > epsilonValue)
		planeParameter[2] = (d2 - planeParameter[0] * x2 - planeParameter[1] * y2) / z2;
	else
		planeParameter[2] = (d3 - planeParameter[0] * x3 - planeParameter[1] * y3) / z3;
}

static inline void computePlaneRANSAC(const std::vector<float>& xs, const std::vector<float>& ys,
	const std::vector<float>& zs, std::vector<float>& coef)
{
	const int num_data = static_cast<int>(xs.size());
	const int num_samples = 10;

	if (num_data < num_samples)
	{
		coef.resize(3);
		coef[0] = 0.f;
		coef[1] = 0.f;
		coef[2] = -1.f;
		return;
	}

	std::vector<int> indices(num_samples);
	int max_score = 0;
	std::vector<float> best_coef(3);
	std::vector<bool> best_flag(num_data, false);

	int max_iteration = 500;
	for (int itr = 0; itr < max_iteration; ++itr)
	{
		int indices_idx = 0;
		std::vector<bool> is_selected(num_data, false);
		while (indices_idx < num_samples)
		{
			int index = getRandom(0, num_data - 1);
			if (is_selected[index]) continue;
			is_selected[index] = true;
			indices[indices_idx] = index;
			++indices_idx;
		}

		cv::Mat1f A(num_samples, 3);
		cv::Mat1f B(num_samples, 1);
		for (int i = 0; i < num_samples; ++i)
		{
			A.ptr<float>(i)[0] = xs[indices[i]];
			A.ptr<float>(i)[1] = ys[indices[i]];
			A.ptr<float>(i)[2] = 1;
			B.ptr<float>(i)[0] = zs[indices[i]];
		}

		cv::Mat1f X = (A.t() * A).inv() * A.t() * B;
		const float a = X.ptr<float>(0)[0];
		const float b = X.ptr<float>(1)[0];
		const float c = X.ptr<float>(2)[0];

		int score = 0;
		std::vector<bool> flag(num_data, false);
		for (int i = 0; i < num_data; ++i)
		{
			const float x = xs[i];
			const float y = ys[i];
			const float z = zs[i];

			const float error = z - (a * x + b * y + c);
			if (error < 1.f)
			{
				++score;
				if (error >= 0.f) flag[i] = true;
			}
		}

		if (max_score < score)
		{
			max_score = score;
			best_coef[0] = a;
			best_coef[1] = b;
			best_coef[2] = c;
			best_flag = flag;
		}

		max_iteration = computeRequiredSamplingNum(num_samples, max_score, num_data, max_iteration, .99f);
	}

	coef = best_coef;

	float sum_xx, sum_yy, sum_xy, sum_x, sum_y, sum_xz, sum_yz, sum_z, n_points;
	sum_xx = sum_yy = sum_xy = sum_x = sum_y =  sum_xz =  sum_yz = sum_z = n_points = 0.f;
	for (int i = 0; i < num_data; ++i)
	{
		if (best_flag[i])
		{
			float x = xs[i], y = ys[i], z = zs[i];
			sum_xx += x * x;
			sum_yy += y * y;
			sum_xy += x * y;
			sum_x  += x;
			sum_y  += y;
			sum_xz += x * z;
			sum_yz += y * z;
			sum_z  += z;
			n_points += 1.f;
		}
	}

	if (n_points <= 10.f) return;
	solvePlaneEquations(sum_xx, sum_xy, sum_x   , sum_xz,
						sum_xy, sum_yy, sum_y   , sum_yz,
						sum_x , sum_y , n_points, sum_z,
						coef);
}

static void fillInvalidated(const cv::Mat3b& I, const cv::Mat1i& C, cv::Mat1w& D)
{
	const int h = I.rows;
	const int w = I.cols;

	auto computePatchCost = [&](int x1, int y1, int x2, int y2) -> int
	{
		const uint c1 = C.ptr<uint>(y1)[x1];
		const uint c2 = C.ptr<uint>(y2)[x2];
		const int cd = HammingDistance(c1, c2);
		return cd;
	};

	int y;
// #pragma omp parallel for
	for (y = 0; y < h; ++y)
	{
		ushort* _D = D.ptr<ushort>(y);
		for (int x = 0; x < w; ++x)
		{
			if (_D[x] != 0) continue;

			int min_norm = 99999;
			ushort d = 0;

			int lx = x, rx = x;
			while (lx >= 0 && _D[lx] == 0) --lx;
			while (rx < w  && _D[rx] == 0) ++rx;
			if (lx >= 0)
			{
				int norm = computePatchCost(x, y, lx, y);
				if (min_norm > norm)
				{
					min_norm = norm;
					d = _D[lx];
				}
			}
			if (rx < w)
			{
				int norm = computePatchCost(x, y, rx, y);
				if (min_norm > norm)
				{
					min_norm = norm;
					d = _D[rx];
				}
			}

			int ty = y, by = y;
			while (ty >= 0 && D.ptr<ushort>(ty)[x] == 0) --ty;
			while (by < h  && D.ptr<ushort>(by)[x] == 0) ++by;
			if (ty >= 0)
			{
				int norm = computePatchCost(x, y, x, ty);
				if (min_norm > norm)
				{
					min_norm = norm;
					d = D.ptr<ushort>(ty)[x];
				}
			}
			if (by < h)
			{
				int norm = computePatchCost(x, y, x, by);
				if (min_norm > norm)
				{
					min_norm = norm;
					d = D.ptr<ushort>(by)[x];
				}
			}

			_D[x] = d;
		}
	}
}

static void fillInvalidated(cv::Mat1w& D)
{
	const int h = D.rows;
	const int w = D.cols;

	int y;
// #pragma omp parallel for
	for (y = 0; y < h; ++y)
	{
		ushort* _D = D.ptr<ushort>(y);
		for (int x = 0; x < w; ++x)
		{
			if (_D[x] != 0) continue;

			int lx = x, rx = x;
			while (lx >= 0 && _D[lx] == 0) --lx;
			while (rx < w  && _D[rx] == 0) ++rx;
			ushort dl = lx >= 0 ? _D[lx] : 9999;
			ushort dr = rx < w  ? _D[rx] : 9999;

			if (lx >= 0 || rx < w) 
			{
				_D[x] = std::min(dl, dr);
			}
		}
	}

// #pragma omp parallel for
	for (y = 0; y < h; ++y)
	{
		ushort* _D = D.ptr<ushort>(y);
		for (int x = 0; x < w; ++x)
		{
			if (_D[x] != 0) continue;

			int ty = y, by = y;
			while (ty >= 0 && D.ptr<ushort>(ty)[x] == 0) --ty;
			while (by < h  && D.ptr<ushort>(by)[x] == 0) ++by;
			float dt = ty >= 0 ? D.ptr<ushort>(ty)[x] : 9999;
			float db = by < h  ? D.ptr<ushort>(by)[x] : 9999;

			if (ty >= 0 || by < h) 
			{
				_D[x] = std::min(dt, db);
			}
		}
	}
}

PatchMatchStereo::PatchMatchStereo()
{}

PatchMatchStereo::~PatchMatchStereo()
{}

void PatchMatchStereo::allocateBufffers(int w, int h)
{
	width_ = w;
	height_ = h;

	MC_[0].create(height_, width_);
	MC_[1].create(height_, width_);

	census_[0].create(height_, width_);
	census_[1].create(height_, width_);
	outlierFlag_[0].create(height_, width_);
	outlierFlag_[1].create(height_, width_);
}

void PatchMatchStereo::destroyBuffers()
{
	census_[0].release();
	census_[1].release();
	outlierFlag_[0].release();
	outlierFlag_[1].release();
	segment_label_image_[0].release();
	segment_label_image_[1].release();
	segmentList_.clear();
	boundaryList_.clear();
	boundaryIndexMatrix_.clear();
}

void PatchMatchStereo::setInputData(const cv::Mat& I1, const cv::Mat& I2, const cv::Mat& D1, const cv::Mat& D2)
{
	D1.copyTo(D_[0]);
	D2.copyTo(D_[1]);
	MC_[0].forEach([&](int& val, const int* p)
	{
		val = computeMatchingCost<0>(I1, I2, p[1], p[0], D_[0].ptr<ushort>(p[0])[p[1]]);
	});
	MC_[1].forEach([&](int& val, const int* p)
	{
		val = computeMatchingCost<1>(I1, I2, p[1], p[0], D_[1].ptr<ushort>(p[0])[p[1]]);
	});

	I1.copyTo(I_corrected_[0]);
	I2.copyTo(I_corrected_[1]);

	cv::Mat G1, G2;
	cv::cvtColor(I1, G1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(I2, G2, cv::COLOR_BGR2GRAY);
	symmetricCensus9x7(G1, census_[0]);
	symmetricCensus9x7(G2, census_[1]);
}

void PatchMatchStereo::LRConsistencyCheck(float max12Diff)
{
	const int h = height_;
	const int w = width_;
	int y;
// #pragma omp parallel for
	for (y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			bool isInvalid1 = false;
			bool isInvalid2 = false;

			const int d1 = D_[0].ptr<ushort>(y)[x];
			const int d2 = D_[1].ptr<ushort>(y)[x];
			isInvalid1 = inBound(x - d1, y, w, h) && std::abs(D_[0].ptr<ushort>(y)[x] - D_[1].ptr<ushort>(y)[x - d1]) > max12Diff;
			isInvalid2 = inBound(x + d2, y, w, h) && std::abs(D_[0].ptr<ushort>(y)[x + d2] - D_[1].ptr<ushort>(y)[x]) > max12Diff;

			if (isInvalid1) D_[0].ptr<ushort>(y)[x] = 0;
			if (isInvalid2) D_[1].ptr<ushort>(y)[x] = 0;
		}
	}
}

template <int VIEW>
float PatchMatchStereo::computeMatchingCost(const cv::Mat& I1, const cv::Mat& I2, int x, int y, int d)
{
	int mx = VIEW == 0 ? x - d : x + d;
	if (!inBound(mx, y, width_, height_)) return 1e+8f;

	const int x1 = VIEW == 0 ? x : mx;
	const int x2 = VIEW == 0 ? mx : x;
	uint c1 = census_[0].ptr<uint>(y)[x1];
	uint c2 = census_[1].ptr<uint>(y)[x2];
	int census_cost = HammingDistance(c1, c2);

	const int d1 = VIEW == 0 ? D_[0].ptr<uint16_t>(y)[x] : D_[1].ptr<uint16_t>(y)[x];
	const int d2 = VIEW == 0 ? D_[1].ptr<uint16_t>(y)[mx] : D_[0].ptr<uint16_t>(y)[mx];
	const int disparity_cost = std::abs(d1 - d2);

	return census_cost + disparity_cost;
}

template <int VIEW>
void PatchMatchStereo::spatialPropagation(const cv::Mat& I1, const cv::Mat& I2)
{
	for (int y = 0; y < height_; ++y)
	{
		for (int x = 0; x < width_; ++x)
		{
			for (int i = 0; i < 4; ++i)
			{
				const int nx = x + dx[i];
				const int ny = y + dy[i];
				if (!inBound(nx, ny, width_, height_)) continue;

				const ushort nd = D_[VIEW].ptr<ushort>(ny)[nx];
				if (nd == 0) continue;
				float cost = computeMatchingCost<VIEW>(I1, I2, nx, ny, nd);
				if (cost < MC_[VIEW].ptr<int>(y)[x])
				{
					D_[VIEW].ptr<ushort>(y)[x] = nd;
					MC_[VIEW].ptr<int>(y)[x] = cost;
				}
			}
		}
	}
}

template <int VIEW>
void PatchMatchStereo::viewPropagation(const cv::Mat& I1, const cv::Mat& I2)
{
	cv::Mat1w& D1 = VIEW == 0 ? D_[0] : D_[1];
	cv::Mat1w& D2 = VIEW == 0 ? D_[1] : D_[0];

	int y;
// #pragma omp parallel for
	for (y = 0; y < height_; ++y)
	{
		for (int x = 0; x < width_; ++x)
		{
			const int d = D1.ptr<ushort>(y)[x];
			const int qx = VIEW == 0 ? x - d : x + d;
			if (!inBound(qx, y, width_, height_)) continue;

			const ushort qd = D2.ptr<ushort>(y)[qx];

			if (qd == 0) continue;
			float cost = computeMatchingCost<VIEW>(I1, I2, x, y, qd);
			if (cost < MC_[VIEW].ptr<int>(y)[x])
			{
				D1.ptr<ushort>(y)[x] = qd;
				MC_[VIEW].ptr<int>(y)[x] = cost;
			}
		}
	}
}

template <int VIEW>
int PatchMatchStereo::appendBoundary(int index1, int index2)
{
	if (boundaryIndexMatrix_[index1][index2] >= 0) return boundaryIndexMatrix_[index1][index2];

	boundaryList_.push_back(PMS::Boundary(index1, index2));
	int newBoundaryIndex = (int)boundaryList_.size() - 1;
	boundaryIndexMatrix_[index1][index2] = newBoundaryIndex;
	boundaryIndexMatrix_[index2][index1] = newBoundaryIndex;

	segmentList_[index1].appendBoundaryIndex(newBoundaryIndex);
	segmentList_[index2].appendBoundaryIndex(newBoundaryIndex);

	return newBoundaryIndex;
}

void PatchMatchStereo::estimateBoundaryLabels()
{
	int numBoundaries = (int)boundaryList_.size();
	for (int boundaryIndex = 0; boundaryIndex < numBoundaries; ++boundaryIndex)
	{
		PMS::Boundary& currentBoundary = boundaryList_[boundaryIndex];
		int segmentIndex1 = currentBoundary.getSegmentIndex(0);
		int segmentIndex2 = currentBoundary.getSegmentIndex(1);
		PMS::Segment& segment1 = segmentList_[segmentIndex1];
		PMS::Segment& segment2 = segmentList_[segmentIndex2];

		float a1 = segment1.getDisparityPlaneParameter(0);
		float b1 = segment1.getDisparityPlaneParameter(1);
		float c1 = segment1.getDisparityPlaneParameter(2);
		float a2 = segment2.getDisparityPlaneParameter(0);
		float b2 = segment2.getDisparityPlaneParameter(1);
		float c2 = segment2.getDisparityPlaneParameter(2);

		float boundaryEnergies[4] = { 0.f };

		float hingeSquaredError = 0.f;
		float hingeError = 0.f;
		hingeError = (a1 - a2) * currentBoundary.getPolynomialCoefficient(3)
				   + (b1 - b2) * currentBoundary.getPolynomialCoefficient(4)
				   + (c1 - c2) * currentBoundary.getPolynomialCoefficient(5);
		hingeSquaredError = currentBoundary.getPolynomialCoefficient(0) * (a1 * a1 + a2 * a2 - 2 * a1 * a2)
						  + currentBoundary.getPolynomialCoefficient(1) * (b1 * b1 + b2 * b2 - 2 * b1 * b2)
						  + currentBoundary.getPolynomialCoefficient(2) * (2 * a1 * b1 + 2 * a2 * b2 - 2 * a1 * b2 -2 * a2 * b1)
						  + currentBoundary.getPolynomialCoefficient(3) * (2 * a1 * c1 + 2 * a2 * c2 - 2 * a1 * c2 -2 * a2 * c1)
						  + currentBoundary.getPolynomialCoefficient(4) * (2 * b1 * c1 + 2 * b2 * c2 - 2 * b1 * c2 -2 * b2 * c1)
						  + currentBoundary.getPolynomialCoefficient(5) * (c1 * c1 + c2 * c2 - 2 * c1 * c2);
		hingeSquaredError /= currentBoundary.getNumBoundaryPixel();
		boundaryEnergies[2] = hingePenalty_ + hingeSquaredError;

		if (hingeError > 0.f)
		{
			boundaryEnergies[0] = occlusionPenalty_;
			boundaryEnergies[1] = occlusionPenalty_ + impossiblePenalty_;
		}
		else
		{
			boundaryEnergies[0] = occlusionPenalty_ + impossiblePenalty_;
			boundaryEnergies[1] = occlusionPenalty_;
		}
		
		float coplanarSquaredError = 0.f;
		coplanarSquaredError += segment1.getPolynomialCoefficientAll(0) * (a1 * a1 + a2 * a2 - 2 * a1 * a2)
							  + segment1.getPolynomialCoefficientAll(1) * (b1 * b1 + b2 * b2 - 2 * b1 * b2)
							  + segment1.getPolynomialCoefficientAll(2) * (2 * a1 * b1 + 2 * a2 * b2 - 2 * a1 * b2 - 2 * a2 * b1)
							  + segment1.getPolynomialCoefficientAll(3) * (2 * a1 * c1 + 2 * a2 * c2 - 2 * a1 * c2 - 2 * a2 * c1)
							  + segment1.getPolynomialCoefficientAll(4) * (2 * b1 * c1 + 2 * b2 * c2 - 2 * b1 * c2 - 2 * b2 * c1)
							  + segment1.getPolynomialCoefficientAll(5) * (c1 * c1 + c2 * c2 - 2 * c1 * c2);
		coplanarSquaredError += segment2.getPolynomialCoefficientAll(0) * (a1 * a1 + a2 * a2 - 2 * a1 * a2)
							  + segment2.getPolynomialCoefficientAll(1) * (b1 * b1 + b2 * b2 - 2 * b1 * b2)
							  + segment2.getPolynomialCoefficientAll(2) * (2 * a1 * b1 + 2 * a2 * b2 - 2 * a1 * b2 - 2 * a2 * b1)
							  + segment2.getPolynomialCoefficientAll(3) * (2 * a1 * c1 + 2 * a2 * c2 - 2 * a1 * c2 - 2 * a2 * c1)
							  + segment2.getPolynomialCoefficientAll(4) * (2 * b1 * c1 + 2 * b2 * c2 - 2 * b1 * c2 - 2 * b2 * c1)
							  + segment2.getPolynomialCoefficientAll(5) * (c1 * c1 + c2 * c2 - 2 * c1 * c2);
		coplanarSquaredError /= (segment1.getNumPixels() + segment2.getNumPixels());
		boundaryEnergies[3] = coplanarSquaredError;

		int minBoundaryLabel = 0;
		if (boundaryEnergies[1] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 1;
		if (boundaryEnergies[2] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 2;
		if (boundaryEnergies[3] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 3;
		currentBoundary.setType(minBoundaryLabel);
	}
}

void PatchMatchStereo::estimateSmoothFitting(int n, int size)
{
// #pragma omp parallel for
	for (int segmentIndex = 0; segmentIndex < n; ++segmentIndex)
	{
		PMS::Segment& currentSegment = segmentList_[segmentIndex];
		int numSegmentPixels = currentSegment.getNumPixels();
		int numDisparityPixels = 0;

		float sumXSqr = currentSegment.getPolynomialCoefficient(0);
		float sumYSqr = currentSegment.getPolynomialCoefficient(1);
		float sumXY   = currentSegment.getPolynomialCoefficient(2);
		float sumX    = currentSegment.getPolynomialCoefficient(3);
		float sumY    = currentSegment.getPolynomialCoefficient(4);
		float sumXD   = currentSegment.getPolynomialCoefficient(5);
		float sumYD   = currentSegment.getPolynomialCoefficient(6);
		float sumD    = currentSegment.getPolynomialCoefficient(7);
		float pointTotal = currentSegment.getPolynomialCoefficient(8);

		numDisparityPixels += (int)currentSegment.getPolynomialCoefficient(8);

		for (int neighborIndex = 0; neighborIndex < currentSegment.getNumBoundaries(); ++neighborIndex)
		{
			int boundaryIndex = currentSegment.getBoundaryIndex(neighborIndex);
			int boundaryLabel = boundaryList_[boundaryIndex].getType();
			if (boundaryLabel < 2) continue;

			PMS::Boundary& currentBoundary = boundaryList_[boundaryIndex];
			int neighborSegmentIndex = currentBoundary.getSegmentIndex(0);
			if (neighborSegmentIndex == segmentIndex) neighborSegmentIndex = currentBoundary.getSegmentIndex(1);
			PMS::Segment& neighborSegment = segmentList_[neighborSegmentIndex];

			if (boundaryLabel == 2)
			{
				int boundaryPixelTotal = currentBoundary.getNumBoundaryPixel();
				float weightValue = smoothRelativeWeight_ / boundaryPixelTotal * size * size;

				sumXSqr += weightValue * currentBoundary.getPolynomialCoefficient(0);
				sumYSqr += weightValue * currentBoundary.getPolynomialCoefficient(1);
				sumXY   += weightValue * currentBoundary.getPolynomialCoefficient(2);
				sumX    += weightValue * currentBoundary.getPolynomialCoefficient(3);
				sumY    += weightValue * currentBoundary.getPolynomialCoefficient(4);
				pointTotal += weightValue * currentBoundary.getPolynomialCoefficient(5);

				sumXD += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * currentBoundary.getPolynomialCoefficient(0)
									  + neighborSegment.getDisparityPlaneParameter(1) * currentBoundary.getPolynomialCoefficient(2)
									  + neighborSegment.getDisparityPlaneParameter(2) * currentBoundary.getPolynomialCoefficient(3));
				sumYD += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * currentBoundary.getPolynomialCoefficient(2)
									  + neighborSegment.getDisparityPlaneParameter(1) * currentBoundary.getPolynomialCoefficient(1)
									  + neighborSegment.getDisparityPlaneParameter(2) * currentBoundary.getPolynomialCoefficient(4));
				sumD  += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * currentBoundary.getPolynomialCoefficient(3)
									  + neighborSegment.getDisparityPlaneParameter(1) * currentBoundary.getPolynomialCoefficient(4)
									  + neighborSegment.getDisparityPlaneParameter(2) * currentBoundary.getPolynomialCoefficient(5));

				numDisparityPixels += (int)currentBoundary.getPolynomialCoefficient(5);
			}
			else
			{
				int neighborSegmentPixelTotal = neighborSegment.getNumPixels();
				float weightValue = smoothRelativeWeight_ / (numSegmentPixels + neighborSegmentPixelTotal) * size * size;

				sumXSqr += weightValue * currentSegment.getPolynomialCoefficientAll(0);
				sumYSqr += weightValue * currentSegment.getPolynomialCoefficientAll(1);
				sumXY   += weightValue * currentSegment.getPolynomialCoefficientAll(2);
				sumX    += weightValue * currentSegment.getPolynomialCoefficientAll(3);
				sumY    += weightValue * currentSegment.getPolynomialCoefficientAll(4);
				pointTotal += weightValue * currentSegment.getPolynomialCoefficientAll(5);

				sumXD += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * currentSegment.getPolynomialCoefficientAll(0)
									  + neighborSegment.getDisparityPlaneParameter(1) * currentSegment.getPolynomialCoefficientAll(2)
									  + neighborSegment.getDisparityPlaneParameter(2) * currentSegment.getPolynomialCoefficientAll(3));
				sumYD += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * currentSegment.getPolynomialCoefficientAll(2)
									  + neighborSegment.getDisparityPlaneParameter(1) * currentSegment.getPolynomialCoefficientAll(1)
									  + neighborSegment.getDisparityPlaneParameter(2) * currentSegment.getPolynomialCoefficientAll(4));
				sumD  += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * currentSegment.getPolynomialCoefficientAll(3)
									  + neighborSegment.getDisparityPlaneParameter(1) * currentSegment.getPolynomialCoefficientAll(4)
									  + neighborSegment.getDisparityPlaneParameter(2) * currentSegment.getPolynomialCoefficientAll(5));

				numDisparityPixels += (int)currentSegment.getPolynomialCoefficientAll(5);

				sumXSqr += weightValue * neighborSegment.getPolynomialCoefficientAll(0);
				sumYSqr += weightValue * neighborSegment.getPolynomialCoefficientAll(1);
				sumXY   += weightValue * neighborSegment.getPolynomialCoefficientAll(2);
				sumX    += weightValue * neighborSegment.getPolynomialCoefficientAll(3);
				sumY    += weightValue * neighborSegment.getPolynomialCoefficientAll(4);
				pointTotal += weightValue * neighborSegment.getPolynomialCoefficientAll(5);

				sumXD += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * neighborSegment.getPolynomialCoefficientAll(0)
									  + neighborSegment.getDisparityPlaneParameter(1) * neighborSegment.getPolynomialCoefficientAll(2)
									  + neighborSegment.getDisparityPlaneParameter(2) * neighborSegment.getPolynomialCoefficientAll(3));
				sumYD += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * neighborSegment.getPolynomialCoefficientAll(2)
									  + neighborSegment.getDisparityPlaneParameter(1) * neighborSegment.getPolynomialCoefficientAll(1)
									  + neighborSegment.getDisparityPlaneParameter(2) * neighborSegment.getPolynomialCoefficientAll(4));
				sumD  += weightValue * (neighborSegment.getDisparityPlaneParameter(0) * neighborSegment.getPolynomialCoefficientAll(3)
									  + neighborSegment.getDisparityPlaneParameter(1) * neighborSegment.getPolynomialCoefficientAll(4)
									  + neighborSegment.getDisparityPlaneParameter(2) * neighborSegment.getPolynomialCoefficientAll(5));

				numDisparityPixels += (int)neighborSegment.getPolynomialCoefficientAll(5);
			}
		}

		if (numDisparityPixels >= 3)
		{
			std::vector<float> planeParameter(3);
			solvePlaneEquations(sumXSqr, sumXY, sumX, sumXD,
								sumXY, sumYSqr, sumY, sumYD,
								sumX, sumY, pointTotal, sumD,
								planeParameter);
			segmentList_[segmentIndex].setDisparityPlane(planeParameter[0], planeParameter[1], planeParameter[2]);
		}
	}
}

template <int VIEW>
void PatchMatchStereo::planeRefinement(const cv::Mat3b& image)
{
	cv::Mat& outlierFlag = outlierFlag_[VIEW];

	// Clustering 
	cv::Mat1w& segment_label_image = segment_label_image_[VIEW];
	cv::Mat1f disparity_image;
	D_[VIEW].convertTo(disparity_image, CV_32F);

	PatchMatchClustering PMC;
	PMC.apply(image, disparity_image);
	// PMC.enforceLabelConnectivity(100);
	PMC.getLabels(segment_label_image);
	num_segments_ = PMC.getNumSegments();

	segmentList_.resize(num_segments_);
	for (auto& s : segmentList_) s.clearConfiguration();

	boundaryList_.clear();
	boundaryIndexMatrix_.resize(num_segments_);
	for (int i = 0; i < num_segments_; ++i)
	{
		boundaryIndexMatrix_[i].resize(num_segments_);
		boundaryIndexMatrix_[i].assign(num_segments_, -1);
	}

	std::vector<std::vector<float>> xs(num_segments_), ys(num_segments_), ds(num_segments_), coefs(num_segments_);
	for (int y = 0; y < height_; ++y)
	{
		ushort* _labels = segment_label_image.ptr<ushort>(y);
		ushort* _D = D_[VIEW].ptr<ushort>(y);

		for (int x = 0; x < width_; ++x)
		{
			if (_D[x] == 0) continue;
			const ushort label = _labels[x];
			xs[label].push_back((float)x);
			ys[label].push_back((float)y);
			ds[label].push_back(_D[x]);
		}
	}
	
	outlierFlag.setTo(0u);
	int i;
// #pragma omp parallel for
	for (i = 0; i < num_segments_; ++i)
	{
		computePlaneRANSAC(xs[i], ys[i], ds[i], coefs[i]);
		segmentList_[i].setDisparityPlane(coefs[i][0], coefs[i][1], coefs[i][2]);
		int num = (int)xs[i].size();
		for (int j = 0; j < num; ++j)
		{
			int x = (int)xs[i][j];
			int y = (int)ys[i][j];
			if (std::abs(segmentList_[i].computeDisparityPlane(x, y) - ds[i][j]) >= 1.f)
				outlierFlag.ptr(y)[x] = 255u;
		}
	}

	for (int y = 0; y < height_; ++y)
	{
		for (int x = 0; x < width_; ++x)
		{
			int index = segment_label_image.ptr<ushort>(y)[x];
			segmentList_[index].appendSegmentPixel(x, y);
			if (D_[VIEW].ptr<ushort>(y)[x] != 0 && outlierFlag.ptr(y)[x] == 0u)
				segmentList_[index].appendSegmentPixelWithDisparity(x, y, D_[VIEW].ptr<ushort>(y)[x]);

			if (x + 1 < width_ && index != segment_label_image.ptr<ushort>(y)[x + 1])
			{
				int neighborSegmentIndex = segment_label_image.ptr<ushort>(y)[x + 1];
				int boundaryIndex = appendBoundary(index, neighborSegmentIndex);
				boundaryList_[boundaryIndex].appendBoundaryPixel(x + 0.5f, (float)y);
			}

			if (y + 1 < height_ && index != segment_label_image.ptr<ushort>(y + 1)[x])
			{
				int neighborSegmentIndex = segment_label_image.ptr<ushort>(y + 1)[x];
				int boundaryIndex = appendBoundary(index, neighborSegmentIndex);
				boundaryList_[boundaryIndex].appendBoundaryPixel((float)x, y + 0.5f);
			}
		}
	}

	for (int itr = 0; itr < 10; ++itr)
	{
		estimateBoundaryLabels();
		estimateSmoothFitting(num_segments_, PMC.getSegmentSideLength());
	}

	const float delta = RefinementPerturbRatio_;
	int y;
// #pragma omp parallel for
	for (y = 0; y < height_; ++y)
	{
		ushort* _labels = segment_label_image.ptr<ushort>(y);
		ushort* _D = D_[VIEW].ptr<ushort>(y);

		for (int x = 0; x < width_; ++x)
		{
			const int label = _labels[x];
			const float plane_d = segmentList_[label].computeDisparityPlane(x, y);
			const float d = _D[x];

			if (std::abs(d - plane_d) >= RefinmentOutlierThreshold_)
				_D[x] = static_cast<uint16_t>(plane_d + 0.5f);
			else
				_D[x] += static_cast<uint16_t>(delta * (plane_d - d) + 0.5f);
		}
	}
}

void PatchMatchStereo::makeSegmentBoundaryData(std::vector<std::vector<int>>& boundaryLabels)
{
	int boundaryTotal = static_cast<int>(boundaryList_.size());
	boundaryLabels.resize(boundaryTotal);
	for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex)
	{
		boundaryLabels[boundaryIndex].resize(3);
		boundaryLabels[boundaryIndex][0] = boundaryList_[boundaryIndex].getSegmentIndex(0);
		boundaryLabels[boundaryIndex][1] = boundaryList_[boundaryIndex].getSegmentIndex(1);
		boundaryLabels[boundaryIndex][2] = boundaryList_[boundaryIndex].getType();
	}
}

void PatchMatchStereo::compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1, cv::Mat& D2)
{
	CV_Assert(I1.type() == CV_8UC3 && D1.type() == CV_16U);
	CV_Assert(I1.type() == I2.type() && D1.type() == D2.type());

	allocateBufffers(I1.cols, I2.rows);
	setInputData(I1, I2, D1, D2);

	LRConsistencyCheck(1.f);
	fillInvalidated(I_corrected_[0], census_[0], D_[0]);
	fillInvalidated(I_corrected_[1], census_[1], D_[1]);

	for (int itr = 0; itr < MCIteration_; ++itr)
	{
		spatialPropagation<0>(I_corrected_[0], I_corrected_[1]);
		spatialPropagation<1>(I_corrected_[0], I_corrected_[1]);
		viewPropagation<0>(I_corrected_[0], I_corrected_[1]);
		viewPropagation<1>(I_corrected_[0], I_corrected_[1]);

		LRConsistencyCheck(1.f);

		if (itr < MCIteration_ - 1)
		{
			fillInvalidated(I_corrected_[0], census_[0], D_[0]);
			fillInvalidated(I_corrected_[1], census_[1], D_[1]);
		}
		else
		{
			fillInvalidated(D_[0]);
			fillInvalidated(D_[1]);
		}
		
		planeRefinement<0>(I_corrected_[0]);
		planeRefinement<1>(I_corrected_[1]);
	}

	cv::Mat disp1, disp2;
	D_[0].convertTo(D1, CV_16U);
	D_[1].convertTo(D2, CV_16U);

	destroyBuffers();
}