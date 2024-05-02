/** @internal
 ** @file     dsift.c
 ** @author   Lazaro Janier Glez Soler
 ** @brief    PHOW (DSIFT) from VLFEAT
 **/

#include <stdio.h>
#include <vl/mathop.h>
#include <vl/dsift.h>
#include <vl/imopv.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "common.h"
#include "phow_dsift.h"

using namespace std;
using namespace cv;

#define ROUND(d) floor(d + 0.5)

float* vl_dsift(float const* img, int rows, int cols, int xy_step, VlDsiftDescriptorGeometry geom, double bounds[], vl_bool useFlatWindow, 
			  double windowSize, bool norm, bool floatDescriptors, int verbose, float contrastthreshold, int* feat_size, int* numKeypoint, double** frames_output)
{   
    /* Input */
    // I -> data, M, N
    // assert I 2D single matrix of Fortran order

    float const *data = (float*) img;
	
	int M = rows;
	int N = cols;   
    
    // step_ -> step
	int step[2] = {xy_step, xy_step};                   

  /* -----------------------------------------------------------------
   *                                                            Do job
   * -------------------------------------------------------------- */
    int numFrames ;
    int descrSize ;
    VlDsiftKeypoint const *frames ;
    float const *descrs ;    

    VlDsiftFilter *dsift ;

    /* note that the image received from MATLAB is transposed */
    dsift = vl_dsift_new (N, M) ;
    vl_dsift_set_geometry(dsift, &geom) ;
    vl_dsift_set_steps(dsift, step[0], step[1]) ;
    
    vl_dsift_set_bounds(dsift,
                          VL_MAX(bounds[1], 0),
                          VL_MAX(bounds[0], 0),
                          VL_MIN(bounds[2], N - 1),
                          VL_MIN(bounds[3], M - 1));
    
	vl_dsift_set_flat_window(dsift, useFlatWindow) ;

    if (windowSize >= 0) 
      vl_dsift_set_window_size(dsift, windowSize) ;

    numFrames = vl_dsift_get_keypoint_num (dsift) ;
    descrSize = vl_dsift_get_descriptor_size (dsift) ;
    geom = *vl_dsift_get_geometry (dsift) ;

	*numKeypoint = numFrames;
	*feat_size = descrSize;

    if (verbose) 
	{
		int stepX ;
		int stepY ;
		int minX ;
		int minY ;
		int maxX ;
		int maxY ;
		vl_bool useFlatWindow ;

		vl_dsift_get_steps (dsift, &stepY, &stepX) ;
		vl_dsift_get_bounds (dsift, &minY, &minX, &maxY, &maxX) ;
		useFlatWindow = vl_dsift_get_flat_window(dsift) ;

		printf("vl_dsift: image size         [W, H] = [%d, %d]\n", N, M) ;
		printf("vl_dsift: bounds:            [minX,minY,maxX,maxY] = [%d, %d, %d, %d]\n",
				minX+1, minY+1, maxX+1, maxY+1) ;
		printf("vl_dsift: subsampling steps: stepX=%d, stepY=%d\n", stepX, stepY) ;
		printf("vl_dsift: num bins:          [numBinT, numBinX, numBinY] = [%d, %d, %d]\n",
				geom.numBinT,
				geom.numBinX,
				geom.numBinY) ;
		printf("vl_dsift: descriptor size:   %d\n", descrSize) ;
		printf("vl_dsift: bin sizes:         [binSizeX, binSizeY] = [%d, %d]\n",
				geom.binSizeX,
				geom.binSizeY) ;
		printf("vl_dsift: flat window:       %s\n", VL_YESNO(useFlatWindow)) ;
		printf("vl_dsift: window size:       %g\n", vl_dsift_get_window_size(dsift)) ;
		printf("vl_dsift: num of features:   %d\n", numFrames) ;
	}

    vl_dsift_process (dsift, data) ;

    frames = vl_dsift_get_keypoints (dsift) ;
    descrs = vl_dsift_get_descriptors (dsift) ;

    /* ---------------------------------------------------------------
     *                                                       Copy back
     * ------------------------------------------------------------ */
	int size_f = 2;//norm?3:2;
        
    //float *tmpDescr = (float *)malloc(sizeof(float) * descrSize) ;
	double* outFrameIter = (double*)malloc(sizeof(double)*size_f*numFrames);
	float*outDescrIter = (float*)malloc(sizeof(float)*descrSize*numFrames);
    for (int k = 0 ; k < numFrames ; ++k) 
	{
		outFrameIter[2*k] = frames[k].x;
		outFrameIter[2*k + 1] = frames[k].y;

		/* We have an implied / 2 in the norm, because of the clipping
			below */
		/*if (norm)
			outFrameIter[k + 2] = frames [k].norm ;*/

		/*vl_dsift_transpose_descriptor (tmpDescr,
										descrs + descrSize * k,
										geom.numBinT,
										geom.numBinX,
										geom.numBinY) ;*/


		if (floatDescriptors) 
		{
			for (int i = 0 ; i < descrSize ; ++i) 
				outDescrIter[k*descrSize + i] = frames[k].norm >= contrastthreshold? VL_MIN(512.0F * descrs[k*descrSize + i], 255.0F): 0 ;
		}
		else
		{
			for (int i = 0 ; i < descrSize ; ++i) 
				outDescrIter[k*descrSize + i] = frames[k].norm >= contrastthreshold? ROUND(VL_MIN(512.0F * descrs[k*descrSize + i], 255.0F)): 0 ;
		}
	}
	//free(outFrameIter);
	vl_dsift_delete (dsift) ;
	*frames_output = outFrameIter;
	return outDescrIter;
}

vector<float> compute_for_image(const Mat image, dsift_extraction_params params)
{	
	int im_width = image.cols;
	int im_height = image.rows;
	vector<float> descr_denseSift;
	vector<float> frames_denseSift;

	
	//convert into float array
	float* imgfloat = (float*)malloc(sizeof(float)*im_width*im_height);
	
	for (int i = 0; i < im_height; i++)
		for (int j = 0; j < im_width; j++)
			imgfloat[i*im_width + j] = (float)image.at<uchar>(i,j)/255;

	VlDsiftDescriptorGeometry geom;
	geom.numBinT = 8; geom.numBinX = 4; geom.numBinY = 4;
	for (int i = 0; i < params.cSize; i++)
	{
		geom.binSizeX = params.binSz[i];
		geom.binSizeY = params.binSz[i];
		//printf("binsize %d\n", params.cSize);
		int off = floor(1 + (float)3 * (params.binSz[params.cSize - 1] - params.binSz[i])/2);
		//vl_dsift_set_bounds(vldf, off, off, im_width - 1, im_height - 1);
		//vl_dsift_set_window_size(vldf, params.windows_size);

		float sigma = (float)params.binSz[i] / params.magnif;
		//sigma = sqrt(pow(sigma, 2) - 0.25);

		//smooth float array image 
		float* img_vec_smooth = (float*)vl_malloc(im_height * im_width * sizeof(float));

		vl_imsmooth_f(img_vec_smooth, im_width, imgfloat, im_width, im_height, im_width, sigma, sigma);

		double bounds[4] = {off - 1, off - 1, im_width - 1, im_height - 1};				
		int feat_size = 0;
		int num_keypoint = 0;
		double* frames = NULL;
		float* descr_scale = vl_dsift(img_vec_smooth, im_height, im_width, params.xy_step, geom, bounds, VL_TRUE, params.windows_size, true, true, false, params.contrastthreshold, &feat_size, &num_keypoint, &frames);		
		descr_denseSift.insert(descr_denseSift.end(), descr_scale, descr_scale + feat_size*num_keypoint);	
		frames_denseSift.insert(frames_denseSift.end(), frames, frames + 2*num_keypoint);
		vl_free(img_vec_smooth);	
		free(descr_scale);
		free(frames);
	}

	free(imgfloat);	
	
	return descr_denseSift;
}

vector<float> compute_for_image(const Mat image, dsift_extraction_params params, vector<float>& frames_denseSift)
{	
	int im_width = image.cols;
	int im_height = image.rows;
	vector<float> descr_denseSift;
	
	//convert into float array
	float* imgfloat = (float*)malloc(sizeof(float)*im_width*im_height);
	
	for (int i = 0; i < im_height; i++)
		for (int j = 0; j < im_width; j++)
			imgfloat[i*im_width + j] = (float)image.at<uchar>(i,j)/255;

	VlDsiftDescriptorGeometry geom;
	geom.numBinT = 8; geom.numBinX = 4; geom.numBinY = 4;
	for (int i = 0; i < params.cSize; i++)
	{
		geom.binSizeX = params.binSz[i];
		geom.binSizeY = params.binSz[i];
		//printf("binsize %d\n", params.cSize);
		int off = floor(1 + (float)3 * (params.binSz[params.cSize - 1] - params.binSz[i])/2);
		//vl_dsift_set_bounds(vldf, off, off, im_width - 1, im_height - 1);
		//vl_dsift_set_window_size(vldf, params.windows_size);

		float sigma = (float)params.binSz[i] / params.magnif;
		//sigma = sqrt(pow(sigma, 2) - 0.25);

		//smooth float array image 
		float* img_vec_smooth = (float*)vl_malloc(im_height * im_width * sizeof(float));

		vl_imsmooth_f(img_vec_smooth, im_width, imgfloat, im_width, im_height, im_width, sigma, sigma);

		double bounds[4] = {off - 1, off - 1, im_width - 1, im_height - 1};				
		int feat_size = 0;
		int num_keypoint = 0;
		double* frames = NULL;
		float* descr_scale = vl_dsift(img_vec_smooth, im_height, im_width, params.xy_step, geom, bounds, VL_TRUE, params.windows_size, true, true, false, params.contrastthreshold, &feat_size, &num_keypoint, &frames);		
		descr_denseSift.insert(descr_denseSift.end(), descr_scale, descr_scale + feat_size*num_keypoint);	
		frames_denseSift.insert(frames_denseSift.end(), frames, frames + 2*num_keypoint);
		vl_free(img_vec_smooth);	
		free(descr_scale);
		free(frames);
	}

	free(imgfloat);	
	
	return descr_denseSift;
}