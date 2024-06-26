/** @internal
 ** @file     phog.c
 ** @author   Lazaro Janier Glez Soler
 ** @brief    PHOG (HOG) from VLFEAT
 **/

#include <stdio.h>
#include <vl/mathop.h>
#include <vl/lbp.h>
#include <vl/imopv.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "plbp.h"
#include "lbp.hpp"
#include "histogram.hpp"

#define magnif 6
#define cSize 4
#define cRadius 3
#define xy_step 3
#define num_patterns 256
static int binSz[cSize] = {4, 8, 16, 32};
static int neighbors[cRadius] = {8, 16, 24};
static int radius[cRadius] = {1, 2, 3};
using namespace std;
using namespace cv;


vector<float> compute_lbp_for_image(const Mat image)
{	
	int im_width = image.cols;
	int im_height = image.rows;
	vector<float> descrs_lbp;
	Mat smooth_img;
	GaussianBlur(image, smooth_img, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
	for (int i = 0; i < cRadius; i++)
	{
		Mat lbp = lbp::ELBP(smooth_img, radius[i], neighbors[i]);
		normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);
		//imshow("lbp", lbp);
		//cv::waitKey(0);
		for (int j = 0; j < cSize; j++)
		{
			Mat sp;
			lbplibrary::spatial_histogram(lbp, sp, num_patterns, binSz[j], binSz[j], xy_step);
			descrs_lbp.assign(sp.begin<int>(), sp.end<int>());
		}
	}
	
	return descrs_lbp;
}

//vector<float> compute_lbp_for_image(const Mat image)
//{
//	int im_width = image.cols;
//	int im_height = image.rows;
//	vector<float> descrs_lbp;
//
//	//convert into float array
//	float* imgfloat = (float*)malloc(sizeof(float)*im_width*im_height);
//
//	for (int i = 0; i < im_height; i++)
//		for (int j = 0; j < im_width; j++)
//			imgfloat[i*im_width + j] = (float)image.at<uchar>(i,j)/255;
//
//
//	for (int i = 0; i < 4; i++)
//	{
//		float sigma = (float)binSz[i] / magnif;
//
//		//smooth float array image
//		float* img_vec_smooth = (float*)vl_malloc(im_height * im_width * sizeof(float));
//
//		vl_imsmooth_f(img_vec_smooth, im_width, imgfloat, im_width, im_height, im_width, sigma, sigma);
//		int num_cols = floorl(im_width/binSz[i]);
//		int num_rows = floorl(im_height/binSz[i]);
//		VlLbp* lbp = vl_lbp_new(VlLbpUniform, false);
//		float* features = (float*)malloc(sizeof(float)*vl_lbp_get_dimension(lbp)*num_cols*num_rows);
//		vl_lbp_process(lbp, features, img_vec_smooth, im_width, im_height, binSz[i]);
//		descrs_lbp.insert(descrs_lbp.end(), features, features + num_cols*num_rows*vl_lbp_get_dimension(lbp));
//		vl_lbp_delete(lbp);
//		vl_free(img_vec_smooth);
//		free(features);
//	}
//	free(imgfloat);
//
//	return descrs_lbp;
//}

