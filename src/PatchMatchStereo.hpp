#ifndef __PATCH_MATCH_STEREO__
#define __PATCH_MATCH_STEREO__

#include <opencv2/core.hpp>
#include <memory>

#include "PatchMatchStruct.hpp"
#include "PatchMatchClustering.hpp"

#define MAX_DISPARITY 256

class PatchMatchStereo
{
public:

	PatchMatchStereo();
	~PatchMatchStereo();
	void compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1, cv::Mat& D2);

	void setIteration(int itr) { MCIteration_ = itr; }//iteration_cost_ = std::vector<float>(itr + 1, 0); }
	void setRefinementPerterbRatio(float r) { RefinementPerturbRatio_ = r; }

private:

	void allocateBufffers(int w, int h);
	void destroyBuffers();
	void setInputData(const cv::Mat& I1, const cv::Mat& I2, const cv::Mat& D1, const cv::Mat& D2);
	void LRConsistencyCheck(float max12Diff = 1.f);
	template <int VIEW = 0> float computeMatchingCost(const cv::Mat& I1, const cv::Mat& I2, int x, int y, int d);
	template <int VIEW = 0> void spatialPropagation(const cv::Mat& I1, const cv::Mat& I2);
	template <int VIEW = 0> void viewPropagation(const cv::Mat& I1, const cv::Mat& I2);
	template <int VIEW = 0> int appendBoundary(int index1, int index2);
	void estimateBoundaryLabels();
	void estimateSmoothFitting(int n, int size);
	template <int VIEW = 0> void planeRefinement(const cv::Mat3b& image);
	void makeSegmentBoundaryData(std::vector<std::vector<int>>& boundaryLabels);

	int width_;
	int height_;

	int num_segments_;

	cv::Mat I_corrected_[2];
	
	cv::Mat1w D_[2];
	cv::Mat1i MC_[2];

	cv::Mat1i census_[2];
	cv::Mat1b outlierFlag_[2];
	cv::Mat1w segment_label_image_[2];

	std::vector<PMS::Segment> segmentList_;
	std::vector<PMS::Boundary> boundaryList_;
	std::vector<std::vector<int>> boundaryIndexMatrix_;

	int   MCIteration_ = 3;
	float RefinementPerturbRatio_ = 0.4f;
	float RefinmentOutlierThreshold_ = 7.f;

	float hingePenalty_ = 0.5f;
	float occlusionPenalty_ = 10.f;
	float impossiblePenalty_ = 30.f;
	float smoothRelativeWeight_ = 0.2f;
};

#endif