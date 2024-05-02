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
#include <vl/imopv.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "psurf.h"

#define cSize 4
#define xy_step 5
static int binSz[cSize] = {4, 8, 16, 32};

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


vector<float> compute_surf_for_image(const Mat image)
{
	vector<float> descr_denseSURF;
	
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
	Ptr<SURF> detector = SURF::create(400,4,3, true, false);
	Mat descriptors;
	//detector->detectAndCompute(image, Mat(), keypoints, descriptors, false);
	detector->compute(image, keypoints, descriptors);
	descr_denseSURF.assign(descriptors.begin<float>(), descriptors.end<float>());

	return descr_denseSURF;
}

