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
#include "porb.h"

#define cSize 4
#define xy_step 7
#define magnif 6
static int binSz[cSize] = { 4, 8, 16, 32 };

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

vector<unsigned char> compute_orb_for_image(const Mat image)
{
	int im_width = image.cols;
	int im_height = image.rows;
	vector<unsigned char> descr_denseORB;

	//sampling points
	vector<KeyPoint> keypoints;
	for (int i = 0; i < cSize; i++)
	{
		for (int k = binSz[i] / 2; k < image.rows - binSz[i] / 2; k += xy_step)
		{
			for (int j = binSz[i] / 2; j < image.cols - binSz[i] / 2; j += xy_step)
				keypoints.push_back(KeyPoint(k, j, binSz[i]));
		}
	}
	//printf("Keypoints %i\n", keypoints.size());
	//computing descriptors
	for (int i = 0; i < cSize; i++)
	{
		Ptr<ORB> detector = ORB::create(500, 1.2f, 8, binSz[i]/2, 0, 2, ORB::HARRIS_SCORE, binSz[i], 20);
		Mat descriptors;
		detector->compute(image, keypoints, descriptors);
		descr_denseORB.insert(descr_denseORB.end(), descriptors.datastart, descriptors.dataend);
	}
	Convert2Binary(descr_denseORB, 8);
	return descr_denseORB;
}

vector<float> compute_orb_for_image_float(const Mat image)
{
	int im_width = image.cols;
	int im_height = image.rows;
	vector<unsigned char> descr_denseORB;

	//sampling points
	vector<KeyPoint> keypoints;
	for (int i = 0; i < cSize; i++)
	{
		for (int k = binSz[i] / 2; k < image.rows - binSz[i] / 2; k += xy_step)
		{
			for (int j = binSz[i] / 2; j < image.cols - binSz[i] / 2; j += xy_step)
				keypoints.push_back(KeyPoint(k, j, binSz[i]));
		}
	}
	//printf("Keypoints %i\n", keypoints.size());
	//computing descriptors
	for (int i = 0; i < cSize; i++)
	{
		Ptr<ORB> detector = ORB::create(500, 1.2f, 8, binSz[i]/2, 0, 2, ORB::HARRIS_SCORE, binSz[i], 20);
		Mat descriptors;
		detector->compute(image, keypoints, descriptors);
		descr_denseORB.insert(descr_denseORB.end(), descriptors.datastart, descriptors.dataend);
	}
	return Convert2BinaryFloat(descr_denseORB, 8);
}
