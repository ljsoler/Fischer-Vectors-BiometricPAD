/** @internal
 ** @file     phog.c
 ** @author   Lazaro Janier Glez Soler
 ** @brief    PHOG (HOG) from VLFEAT
 **/

#include <stdio.h>
#include <vl/mathop.h>
#include <vl/hog.h>
#include <vl/imopv.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "phog.h"

#define magnif 6
#define numOfOrientations 9
#define cSize 4
static int binSz[cSize] = {4, 8, 16, 32};
using namespace std;
using namespace cv;

vector<float> compute_phog_for_image(const Mat image)
{
	int im_width = image.cols;
	int im_height = image.rows;
	vector<float> descrs_hog;

	//convert into float array
	float* imgfloat = (float*)malloc(sizeof(float)*im_width*im_height);

	for (int i = 0; i < im_height; i++)
		for (int j = 0; j < im_width; j++)
			imgfloat[i*im_width + j] = (float)image.at<uchar>(i,j)/255;

	for (int i = 0; i < cSize; i++)
	{
		float sigma = (float)binSz[i] / magnif;

		//smooth float array image
		float* img_vec_smooth = (float*)vl_malloc(im_height * im_width * sizeof(float));

		vl_imsmooth_f(img_vec_smooth, im_width, imgfloat, im_width, im_height, im_width, sigma, sigma);

		VlHog * hog = vl_hog_new(VlHogVariantDalalTriggs, numOfOrientations, VL_FALSE);
		vl_hog_put_image(hog, img_vec_smooth, im_width, im_height, 1, binSz[i]) ;
		vl_size hogWidth = vl_hog_get_width(hog);
		vl_size hogHeight = vl_hog_get_height(hog);
		vl_size hogDimenison = vl_hog_get_dimension(hog) ;
		float* hogArray = (float*)malloc(hogWidth*hogHeight*hogDimenison*sizeof(float));
		vl_hog_extract(hog, hogArray);
		vl_hog_delete(hog);

		descrs_hog.insert(descrs_hog.end(), hogArray, hogArray + hogWidth*hogHeight*hogDimenison);
		vl_free(img_vec_smooth);
		free(hogArray);
	}
	free(imgfloat);

	return descrs_hog;
}

