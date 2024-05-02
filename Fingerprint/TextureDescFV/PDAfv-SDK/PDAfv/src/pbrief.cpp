/** @internal
 ** @file     phog.c
 ** @author   Lazaro Janier Glez Soler
 ** @brief    PHOG (HOG) from VLFEAT
 **/

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vl/imopv.h>
#include "pbrief.h"

#define cSize 4
#define xy_step 7
#define magnif 6
#define featureScaleLevels 5
static int binSz[cSize] = { 4, 8, 16, 32 };


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


vector<unsigned char> compute_brief_for_image(const Mat image)
{
	int im_width = image.cols;
	int im_height = image.rows;
	vector<unsigned char> descr_denseBRIEF;

	//sampling points
	vector<KeyPoint> keypoints;
	float curScale = static_cast<float>(1.f);
	int curStep = xy_step;
	int curBound = 0;
	for (int curLevel = 0; curLevel < featureScaleLevels; curLevel++)
	{
		for (int x = curBound; x < image.cols - curBound; x += curStep)
		{
			for (int y = curBound; y < image.rows - curBound; y += curStep)
			{
				keypoints.push_back(KeyPoint(static_cast<float>(x), static_cast<float>(y), curScale, -1, 0.0f, curLevel));
			}
		}
		curScale = static_cast<float>(curScale * sqrtf(2.0));
	}
	//printf("Keypoints %i\n", keypoints.size());
	//computing descriptors
	for (int i = 0; i < cSize; i++)
	{
		Ptr<BriefDescriptorExtractor> detector = BriefDescriptorExtractor::create();
		Mat descriptors;
		detector->compute(image, keypoints, descriptors);
		descr_denseBRIEF.insert(descr_denseBRIEF.end(), descriptors.datastart, descriptors.dataend);
	}
	Convert2Binary(descr_denseBRIEF, 8);
	return descr_denseBRIEF;
}

vector<float> compute_brief_for_image_float(const Mat image)
{
	int im_width = image.cols;
	int im_height = image.rows;
	vector<unsigned char> descr_denseBRIEF;

	//sampling points
	vector<KeyPoint> keypoints;
	float curScale = static_cast<float>(1.f);
	int curStep = xy_step;
	int curBound = 0;
	for (int curLevel = 0; curLevel < featureScaleLevels; curLevel++)
	{
		for (int x = curBound; x < image.cols - curBound; x += curStep)
		{
			for (int y = curBound; y < image.rows - curBound; y += curStep)
			{
				keypoints.push_back(KeyPoint(static_cast<float>(x), static_cast<float>(y), curScale, -1, 0.0f, curLevel));
			}
		}
		curScale = static_cast<float>(curScale * sqrtf(2.0));
	}
	//printf("Keypoints %i\n", keypoints.size());
	//computing descriptors
	for (int i = 0; i < cSize; i++)
	{
		Ptr<BriefDescriptorExtractor> detector = BriefDescriptorExtractor::create();
		Mat descriptors;
		detector->compute(image, keypoints, descriptors);
		descr_denseBRIEF.insert(descr_denseBRIEF.end(), descriptors.datastart, descriptors.dataend);
	}
	return Convert2BinaryFloat(descr_denseBRIEF, 8);
}
