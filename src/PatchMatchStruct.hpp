#ifndef __PATCH_MATCH_STRUCT__
#define __PATCH_MATCH_STRUCT__

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <memory>

// struct PlaneParameter
// {
// 	float nx, ny, nz;
// 	float a, b, c;
// 	float compute(int x, int y) { return a * x + b * y + c; }
// };

// class PlaneParamMatrix
// {
// public:

// 	PlaneParamMatrix() : w(0), h(0), data(nullptr) {}
// 	PlaneParamMatrix(int width, int height) { create(width, height); }

// 	void create(int width, int height)
// 	{
// 		w = width;
// 		h = height;
// 		data = std::make_unique<PlaneParameter[]>(w * h);
// 	}

// 	void setParameter(int x, int y, float d, float nx, float ny, float nz)
// 	{
// 		data[w * y + x].nx = nx;
// 		data[w * y + x].ny = ny;
// 		data[w * y + x].nz = nz;
// 		data[w * y + x].a = -nx / nz;
// 		data[w * y + x].b = -ny / nz;
// 		data[w * y + x].c = (nx * x + ny * y + nz * d) / nz;
// 	}

// 	void resetParameter(int x, int y, float d)
// 	{
// 		data[w * y + x].c = (data[w * y + x].nx * x + data[w * y + x].ny * y + data[w * y + x].nz * d) / data[w * y + x].nz;
// 	}

// 	float getValue(int x, int y)
// 	{ 
// 		return data[w * y + x].compute(x, y);
// 	}

// private:

// 	int w, h;
// 	std::unique_ptr<PlaneParameter[]> data;
// };


namespace PMS
{

// struct DispPixel
// {
// 	float nx_, ny_, nz_;
// 	float a_, b_, c_;
// 	bool valid;

// 	DispPixel() : nx_(0), ny_(0), nz_(0), a_(0), b_(0), c_(0), valid(false) {}

// 	void set(float nx, float ny, float nz, float d, int x, int y)
// 	{
// 		if (d > 0.f)
// 		{
// 			nx_ = nx;
// 			ny_ = ny;
// 			nz_ = nz;
// 			a_ = -nx / nz;
// 			b_ = -ny / nz;
// 			c_ = (nx * x + ny * y + nz * d) / nz;
// 			valid = true;
// 		}
// 		else
// 		{
// 			valid = false;
// 		}
// 	}

// 	void reset(float d, int x, int y)
// 	{
// 		a_ = -nx_ / nz_;
// 		b_ = -ny_ / nz_;
// 		c_ = (nx_ * x + ny_ * y + nz_ * d) / nz_;
// 		valid = true;
// 	}

// 	void set(const DispPixel& f)
// 	{
// 		nx_ = f.nx_;
// 		ny_ = f.ny_;
// 		nz_ = f.nz_;
// 		a_ = f.a_;
// 		b_ = f.b_;
// 		c_ = f.c_;
// 		valid = f.valid;
// 	}

// 	float getDisparity(int x, int y) const
// 	{
// 		return valid ? std::max(a_ * x + b_ * y + c_, 0.f) : 0.f;
// 	}
// };

// struct DispImg
// {
// 	int w, h;
// 	DispPixel* data;
// 	float* cost;
// 	void create(int width, int height) {
// 		w = width, h = height;
// 		data = reinterpret_cast<DispPixel*>(malloc(w * h * sizeof(DispPixel)));
// 		cost = reinterpret_cast<float*>(malloc(w * h * sizeof(float)));
// 		for (int i = 0; i < width * height; ++i) cost[i] = 1e+8f;
// 	}
// 	void destroy() { free(data); free(cost); }
// 	void createDisparityImage(cv::Mat& disparity) {
// 		disparity.create(h, w, CV_32F);
// 		disparity.forEach<float>([&](float& val, const int* p) {
// 			int x = p[1], y = p[0];
// 			float d = data[w * y + x].getDisparity(x, y);
// 			if (d < 0.f) d = 0.f;
// 			val = d;
// 		});
// 	}
// 	void show(const char* window_name) {
// 		cv::Mat disparity;
// 		createDisparityImage(disparity);
// 		disparity.convertTo(disparity, CV_8U, 256. / 128);
// 		cv::applyColorMap(disparity, disparity, cv::COLORMAP_JET);
// 		cv::imshow(window_name, disparity);
// 	}
// };

struct Segment
{
	int numPixels;
	float positionSum[2];
	float disparityPlane[3];
	std::vector<int> neighborSegmentIndices;
	std::vector<int> boundaryIndices;
	float polynomialCoefficients[9];
	float polynomialCoefficientsAll[6];

	Segment() {
		clear();
	}
	~Segment() {
		neighborSegmentIndices.clear();
		boundaryIndices.clear();
	}
	void clear() {
		numPixels = 0;
		positionSum[0] = positionSum[1] = 0.f;
		disparityPlane[0] = disparityPlane[1] = 0.f, disparityPlane[2] = -1.f;
		neighborSegmentIndices.clear();
		boundaryIndices.clear();
		for (int i = 0; i < 9; ++i) polynomialCoefficients[i] = 0.f;
		for (int i = 0; i < 6; ++i) polynomialCoefficientsAll[i] = 0.f;
	}
	void clearConfiguration() {
		numPixels = 0;
		positionSum[0] = positionSum[1] = 0.f;
		neighborSegmentIndices.clear();
		boundaryIndices.clear();
		for (int i = 0; i < 9; ++i) polynomialCoefficients[i] = 0.f;
		for (int i = 0; i < 6; ++i) polynomialCoefficientsAll[i] = 0.f;
	}
	void setDisparityPlane(float a, float b, float c) { disparityPlane[0] = a, disparityPlane[1] = b, disparityPlane[2] = c; }
	void appendBoundaryIndex(int boundaryIndex) { boundaryIndices.push_back(boundaryIndex); }
	void appendSegmentPixel(int x, int y) {
		polynomialCoefficientsAll[0] += x * x;
		polynomialCoefficientsAll[1] += y * y;
		polynomialCoefficientsAll[2] += x * y;
		polynomialCoefficientsAll[3] += x;
		polynomialCoefficientsAll[4] += y;
		polynomialCoefficientsAll[5] += 1.f;
	}
	void appendSegmentPixelWithDisparity(int x, int y, float d) {
		polynomialCoefficients[0] += x * x;
		polynomialCoefficients[1] += y * y;
		polynomialCoefficients[2] += x * y;
		polynomialCoefficients[3] += x;
		polynomialCoefficients[4] += y;
		polynomialCoefficients[5] += x * d;
		polynomialCoefficients[6] += y * d;
		polynomialCoefficients[7] += d;
		polynomialCoefficients[8] += 1.f;
	}
	float getPolynomialCoefficient(int index) { return polynomialCoefficients[index]; }
	float getPolynomialCoefficientAll(int index) { return polynomialCoefficientsAll[index]; }
	float getDisparityPlaneParameter(int index) { return disparityPlane[index]; }
	int getNumPixels() { return (int)polynomialCoefficientsAll[5]; }
	int getNumBoundaries() { return (int)boundaryIndices.size(); }
	int getBoundaryIndex(int index) { return boundaryIndices[index]; }
	float computeDisparityPlane(int x, int y) { return disparityPlane[0] * x + disparityPlane[1] * y + disparityPlane[2]; }
	bool hasDisparityPlane() { return !(disparityPlane[0] == 0.f && disparityPlane[1] == 0.f && disparityPlane[2] == -1.f); }
};

struct Boundary
{
	int type;
	int segmentIndices[2];
	float polynomialCoefficients[6];

	Boundary() { segmentIndices[0] = segmentIndices[1] = -1; clearCoefficients(); }
	Boundary(int index1, int index2) {
		if (index1 < index2) segmentIndices[0] = index1, segmentIndices[1] = index2;
		else segmentIndices[0] = index2, segmentIndices[1] = index1;
		clearCoefficients();
	}
	void clear() {
		segmentIndices[0] = segmentIndices[1] = -1;
	}
	void clearCoefficients() { for (int i = 0; i < 6; ++i) polynomialCoefficients[i] = 0.f; }
	void setType(int Type_) { type = Type_; }
	void appendBoundaryPixel(float x, float y) {
		polynomialCoefficients[0] += x * x;
		polynomialCoefficients[1] += y * y;
		polynomialCoefficients[2] += x * y;
		polynomialCoefficients[3] += x;
		polynomialCoefficients[4] += y;
		polynomialCoefficients[5] += 1.f;
	}
	int getType() { return type; }
	int getSegmentIndex(int index) { return segmentIndices[index]; }
	bool isConsistOf(int index1, int index2) {
		return (segmentIndices[0] == index1 && segmentIndices[1] == index2)
				|| (segmentIndices[0] == index2 && segmentIndices[1] == index1);
	}
	int isInclude(int segmentIndex) {
		if (segmentIndex == segmentIndices[0]) return 0;
		if (segmentIndex == segmentIndices[1]) return 1;
		return -1;
	}
	int getNumBoundaryPixel() { return (int)polynomialCoefficients[5]; }
	float getPolynomialCoefficient(int index) { return polynomialCoefficients[index]; }
};

}

#endif