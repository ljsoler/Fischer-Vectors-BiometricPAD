/** @internal
 ** @file     pbsif.c
 ** @author   Lazaro Janier Gonzalez-Soler
 ** @brief    PHOG (HOG) from VLFEAT
 **/

#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "pbsif.h"
#include "histogram.h"

//#define filter_config_path "D:/Work/Projects/PAD_projects/vc-scripts/PAD_Workspace/FacePADfV_SDK/FacePADfV_SDK/filter_config.txt"
//#define filter_multiscale_config_path "../FacePADfV_SDK/filter_multiS_config.txt"
#define cSize 4
#define xy_step 3
#define nBins 128
static int binSz[cSize] = { 4, 6, 8, 10 };

using namespace std;
using namespace cv;

int read_filter(int filter_dims[], float** filter_data, const char* filter_path) {

	FILE* file = fopen(filter_path, "rb+");
	char* fpath = (char*)malloc(sizeof(char*)*256);
	fscanf(file, "%s\n", fpath);
	fclose(file);
    //read filter-----------------------------	
	string path = string(fpath);
	std::ifstream infile;
	infile.open(path.c_str(), std::ios::in | std::ios::binary);
	int dim = 0, f_size = 0;
	infile.read((char*)&dim, sizeof(int));
	infile.read((char*)&f_size, sizeof(int));
	float* copied_data = (float*)malloc(sizeof(float)*dim*dim*f_size);
	infile.read((char*)copied_data, dim*dim*f_size*sizeof(float));
	infile.close();
	//printf("%f\n", copied_data[0]);
    //-copy dims-------------------------------
    filter_dims[0] = dim;
    filter_dims[1] = dim;
	filter_dims[2] = f_size;

    //-copy filter-data------------------------    
    *filter_data = copied_data;    
	free(fpath);
    return EXIT_SUCCESS;
}

//int read_several_filter(int** filter_dims[], float*** filter_data, int* num_filters) {
//
//	FILE* file = fopen(filter_multiscale_config_path, "rb+");
//	int nFilter = 0;
//	fscanf(file, "%i\n", &nFilter);
//	char** fpath = (char**)malloc(sizeof(char*)*nFilter);
//	for(int i = 0; i < nFilter; i++)
//	{
//		fpath[i] = (char*)malloc(sizeof(char*)*256);
//		fscanf(file, "%s\n", fpath[i]);
//	}
//	fclose(file);
//    //read filter-----------------------------
//	*num_filters = nFilter;
//	float** copied_data = (float**)malloc(sizeof(float*)*nFilter);
//	int** fdim = (int**)malloc(sizeof(int*)*nFilter);
//	for(int i = 0; i < nFilter; i++)
//	{
//		string path = string(fpath[i]);
//		std::ifstream infile;
//		infile.open(path.c_str(), std::ios::in | std::ios::binary);
//		int dim = 0, f_size = 0;
//		infile.read((char*)&dim, sizeof(int));
//		infile.read((char*)&f_size, sizeof(int));
//		copied_data[i] = (float*)malloc(sizeof(float)*dim*dim*f_size);
//		infile.read((char*)copied_data[i], dim*dim*f_size*sizeof(float));
//		infile.close();
//		//printf("%f\n", copied_data[0]);
//		//-copy dims-------------------------------
//		fdim[i] = (int*)malloc(sizeof(int)*3);
//		fdim[i][0] = dim;
//		fdim[i][1] = dim;
//		fdim[i][2] = f_size;
//		free(fpath[i]);
//	}
//    //-copy filter-data------------------------
//    *filter_data = copied_data;
//    *filter_dims = fdim;
//	free(fpath);
//    return EXIT_SUCCESS;
//}

vector<float> compute_bsif_for_image(const Mat image, char* filter_path)
{	
	/*imshow("Original", image);
	cv::waitKey(0);*/
	int im_width = image.cols;
	int im_height = image.rows;
	vector<float> descrs_bsif;	
	
	int filter_dim[3];
	float* filter_data = NULL;
	//Reading filters
	//printf("Reading filter\n");
	read_filter(filter_dim, &filter_data, filter_path);
	int num_filter = filter_dim[2]; //Number of filters
	int r = floor(filter_dim[0]/2);

	unsigned int* code = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * im_height * im_width));
	
	for (int i = 0; i < im_height*im_width; i++) code[i] = 1;
	//Building bsif code
	//printf("Building bsif code\n");
	for (int i = 0; i < num_filter; i++)
	{
		Mat out;		
		Mat kernel(filter_dim[0], filter_dim[1], CV_32F, &filter_data[(filter_dim[2] - i - 1)*filter_dim[0]*filter_dim[1]]);
		float b = kernel.at<float>(2,2);
		cv::filter2D(image, out, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
		for (int j = 0; j < im_height; j++)
			for (int k = 0; k < im_width; k++)
				code[j*im_width + k] += (out.data[j*im_width + k] > 0)? pow(2, i): 0;
	}	
	if(filter_data != NULL) free(filter_data);
	//Building histogram
	/*Mat bsif = Mat(im_height, im_width, CV_16UC1, code);
	ushort aux = bsif.at<ushort>(0, 0);
	normalize(bsif, bsif, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("BSIF", bsif);
	cv::waitKey(0);*/

	Geometry geom;
	geom.numBinX = 4; geom.numBinY = 4;
	int feat_size = nBins;

	int binSize = pow(2, num_filter) > nBins? pow(2, num_filter) / nBins: 1;

	for(int i = 0; i < cSize; i++)
	{
		geom.binSizeX = binSz[i];
		geom.binSizeY = binSz[i];
		//printf("binsize %d\n", params.cSize);
		int off = floor(1 + (float)3 * (binSz[cSize - 1] - binSz[i]) / 2);

		double bounds[4] = { off - 1, off - 1, im_width - 1, im_height - 1 };
		int numKeypoint = 0;

		float* descr = histogram::calcSpatialHistogram(code, im_width, im_height, nBins, binSize, bounds, geom, xy_step, xy_step, true, true, &numKeypoint);
		descrs_bsif.insert(descrs_bsif.end(), descr, descr + feat_size * numKeypoint);

		free(descr);
	}

	free(code);

	return descrs_bsif;
}

vector<float> compute_bsif_for_image(const Mat image, int filter_dim[], float* filter_data)
{
	/*imshow("Original", image);
	cv::waitKey(0);*/
	int im_width = image.cols;
	int im_height = image.rows;
	vector<float> descrs_bsif;

	int num_filter = filter_dim[2]; //Number of filters
	int r = floor(filter_dim[0] / 2);

	unsigned int* code = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * im_height * im_width));

	for (int i = 0; i < im_height*im_width; i++) code[i] = 1;
	//Building bsif code
	//printf("Building bsif code\n");
	for (int i = 0; i < num_filter; i++)
	{
		Mat out;
		Mat kernel(filter_dim[0], filter_dim[1], CV_32F, &filter_data[(filter_dim[2] - i - 1)*filter_dim[0] * filter_dim[1]]);
		float b = kernel.at<float>(2, 2);
		cv::filter2D(image, out, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
		for (int j = 0; j < im_height; j++)
			for (int k = 0; k < im_width; k++)
				code[j*im_width + k] += (out.data[j*im_width + k] > 0) ? pow(2, i) : 0;
	}
	//Building histogram
	/*Mat bsif = Mat(im_height, im_width, CV_16UC1, code);
	ushort aux = bsif.at<ushort>(0, 0);
	normalize(bsif, bsif, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("BSIF", bsif);
	cv::waitKey(0);*/

	Geometry geom;
	geom.numBinX = 4; geom.numBinY = 4;
	int feat_size = nBins;

	int binSize = pow(2, num_filter) > nBins ? pow(2, num_filter) / nBins : 1;

	for (int i = 0; i < cSize; i++)
	{
		geom.binSizeX = binSz[i];
		geom.binSizeY = binSz[i];
		//printf("binsize %d\n", params.cSize);
		int off = floor(1 + (float)3 * (binSz[cSize - 1] - binSz[i]) / 2);

		double bounds[4] = { off - 1, off - 1, im_width - 1, im_height - 1 };
		int numKeypoint = 0;

		float* descr = histogram::calcSpatialHistogram(code, im_width, im_height, nBins, binSize, bounds, geom, xy_step, xy_step, true, true, &numKeypoint);
		descrs_bsif.insert(descrs_bsif.end(), descr, descr + feat_size * numKeypoint);

		free(descr);
	}

	free(code);

	return descrs_bsif;
}

vector<float> compute_bsif_for_image(const Mat image, int filter_dim[], float* filter_data, vector<tuple<int, int>> points)
{
	/*imshow("Original", image);
	cv::waitKey(0);*/
	int im_width = image.cols;
	int im_height = image.rows;
	vector<float> descrs_bsif;

	int num_filter = filter_dim[2]; //Number of filters
	int r = floor(filter_dim[0] / 2);

	unsigned int* code = static_cast<unsigned int*>(malloc(sizeof(unsigned int) * im_height * im_width));

	for (int i = 0; i < im_height*im_width; i++) code[i] = 1;
	//Building bsif code
	//printf("Building bsif code\n");
	for (int i = 0; i < num_filter; i++)
	{
		Mat out;
		Mat kernel(filter_dim[0], filter_dim[1], CV_32F, &filter_data[(filter_dim[2] - i - 1)*filter_dim[0] * filter_dim[1]]);
		float b = kernel.at<float>(2, 2);
		cv::filter2D(image, out, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
		for (int j = 0; j < im_height; j++)
			for (int k = 0; k < im_width; k++)
				code[j*im_width + k] += (out.data[j*im_width + k] > 0) ? pow(2, i) : 0;
	}
	//Building histogram
	/*Mat bsif = Mat(im_height, im_width, CV_16UC1, code);
	ushort aux = bsif.at<ushort>(0, 0);
	normalize(bsif, bsif, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("BSIF", bsif);
	cv::waitKey(0);*/

	Geometry geom;
	geom.numBinX = 4; geom.numBinY = 4;
	int feat_size = nBins;

	int binSize = pow(2, num_filter) > nBins ? pow(2, num_filter) / nBins : 1;

	int* valid_points = NULL;

	if(points.size() > 0)
		valid_points = (int*)calloc(points.size(), sizeof(int));

	for (int i = 0; i < cSize; i++)
	{
		geom.binSizeX = binSz[i];
		geom.binSizeY = binSz[i];
		//printf("binsize %d\n", params.cSize);
		int off = floor(1 + (float)3 * (binSz[cSize - 1] - binSz[i]) / 2);

		double bounds[4] = { off - 1, off - 1, im_width - 1, im_height - 1 };
		int numKeypoint = 0;


		float* descr = (points.size() == 0)?
				histogram::calcSpatialHistogram(code, im_width, im_height, nBins, binSize, bounds, geom, xy_step, xy_step, true, true, &numKeypoint)
				: histogram::calcSpatialHistogramForFixedPoints(code, im_width, im_height, nBins, binSize, geom, true, true, points, &numKeypoint, valid_points);

		descrs_bsif.insert(descrs_bsif.end(), descr, descr + feat_size * numKeypoint);

		free(descr);
	}

	free(code);

	return descrs_bsif;
}

//vector<float> compute_multiscale_bsif_for_image(const Mat image)
//{
//	int im_width = image.cols;
//	int im_height = image.rows;
//	vector<float> descrs_bsif;
//
//	int** filter_dim;
//	float** filter_data = NULL;
//	int nFilter = 0;
//	//Reading filters
//	read_several_filter(&filter_dim, &filter_data, &nFilter);
//	//int num_filter = filter_dim[2]; //Number of filters
//	//int r = floor(filter_dim[0]/2);
//	Mat smooth_img;
//	GaussianBlur(image, smooth_img, Size(5,5), 1.5, 1.5, BORDER_CONSTANT);
//
//	//Building bsif code
//	for (int j = 0; j < nFilter; j++)
//	{
//		unsigned short* code = (unsigned short*)malloc(sizeof(unsigned short)*im_height*im_width);
//		for (int i = 0; i < im_height*im_width; i++) code[i] = 1;
//		for (int i = 0; i < filter_dim[j][2]; i++)
//		{
//			Mat out;
//			Mat kernel(filter_dim[j][0], filter_dim[j][1], CV_32F, &filter_data[j][(filter_dim[j][2] - i - 1)*filter_dim[j][0]*filter_dim[j][1]]);
//			float b = kernel.at<float>(0,0);
//			filter2D(smooth_img, out, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
//			for (int t = 0; t < im_height; t++)
//				for (int k = 0; k < im_width; k++)
//					code[t*im_width + k] += (out.data[t*im_width + k] > 0)? pow(2, i): 0;
//		}
//		//Building histogram
//		Mat bsif = Mat(im_height, im_width, CV_16UC1, code);
////		normalize(bsif, bsif, 0, 255, NORM_MINMAX, CV_8UC1);
////		imshow("BSIF", bsif);
////		cv::waitKey(0);
//		/*for(int i = 0; i < cSize; i++)
//		{
//			Mat sp;
//			lbplibrary::spatial_histogram(bsif, sp, pow(2, filter_dim[j][2]), binSz[i], binSz[i], xy_step);
//			descrs_bsif.insert(descrs_bsif.end(), sp.begin<int>(), sp.end<int>());
//		}
//		if(filter_data[j] != NULL) free(filter_data[j]); free(filter_dim[j]);*/
//		free(code);
//	}
//	free(filter_data);
//	free(filter_dim);
//
//	return descrs_bsif;
//}
//
//vector<unsigned char> compute_multiscale_binary_bsif_for_image(const Mat image)
//{
//	int im_width = image.cols;
//	int im_height = image.rows;
//	vector<unsigned char> descrs_bsif;
//
//	int** filter_dim;
//	float** filter_data = NULL;
//	int nFilter = 0;
//	//Reading filters
//	read_several_filter(&filter_dim, &filter_data, &nFilter);
//	//int num_filter = filter_dim[2]; //Number of filters
//	//int r = floor(filter_dim[0]/2);
//	Mat smooth_img;
//	GaussianBlur(image, smooth_img, Size(5,5), 1.5, 1.5, BORDER_CONSTANT);
//
//	//Building bsif code
//	unsigned short* code = (unsigned short*)malloc(sizeof(unsigned short)*im_height*im_width*filter_dim[0][2]);
//	for (int j = 0; j < nFilter; j++)
//	{
//		for (int i = 0; i < filter_dim[j][2]; i++)
//		{
//			Mat out;
//			Mat kernel(filter_dim[j][0], filter_dim[j][1], CV_32F, &filter_data[j][(filter_dim[j][2] - i - 1)*filter_dim[j][0]*filter_dim[j][1]]);
//			float b = kernel.at<float>(0,0);
//			filter2D(smooth_img, out, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
//			for (int t = 0; t < im_height; t++)
//				for (int k = 0; k < im_width; k++)
//					code[filter_dim[j][2]*(t*im_width + k) + j] = (out.data[t*im_width + k] > 0)? 1: 0;
//		}
//	}
//	descrs_bsif.insert(descrs_bsif.end(), code, code + im_height*im_width*filter_dim[0][2]);
//	free(code);
//	free(filter_data);
//	free(filter_dim);
//
//	return descrs_bsif;
//}


