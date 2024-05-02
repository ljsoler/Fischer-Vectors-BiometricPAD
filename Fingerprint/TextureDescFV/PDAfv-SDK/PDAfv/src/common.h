#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <string.h>
#include "math.h" 
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

#define name_size 50

template<class T>
struct extract_params
{	
	int mode;
	vector<T> (*feature_extractor)(const Mat);
	char* imageDir;
	char* outputDir;	
	int num_cluster;
	int features_dim;	
	int pca_dim;		
	int descriptors_count;	
	string database_name;
	bool PCA_usage;
	double threshold;
};

void Convert2Binary(vector<uchar> &x, int bits);

vector<float> Convert2BinaryFloat(vector<uchar> x, int bits);

Mat image_segmentation(Mat src);

int SaveDescrBinary(string path, const unsigned char* descr, long descr_size);

int SaveDescrBinary(string path, const float* descr, long descr_size);

int SaveDescrBinary(string path, float* descr, long descr_size);

int LoadDescrBinary(string path, unsigned char* descr, long size);

int LoadDescrBinary(string path, float* descr, long size);

int LoadDescrBinaryInfo(string path, long *size);

int LoadTextFile(string path, vector<string> &output);

int SaveDescrAsText(string path, vector<float> descr, int dim_desc);

int SaveDescrAsText(string path, vector<float> descr);

int loadDBInfo(string path, int* num_images);

int SaveDBInfo(string path, int num_images);

int SaveDescrChunkBinary(string path, float* descr, long descr_size);

double* GetLabels(vector<string> image_url, const char* positive_class);

double* GetLabels(vector<string> image_url, const char* positive_class, int* fake_count, int* live_count);

void copyFile(const std::string& fileNameFrom, const std::string& fileNameTo);

Mat readImage(string image_path);

int load_model(string path, float* model[], float bias[], int* model_size, int nr_classes);

int save_model(string path, float* model[], float* bias, int model_size, int nr_classes);

int get_files(const std::string folder_path, const std::string extension, std::vector<std::string> &files, int max_number = INT_MAX, const bool deeper_search = true);

bool is_file_exist(const char *fileName);

string to_string_(int value);

double Min(int ndim, const double* const a);

int ArgMin(int ndim, const double* const a);

int ArgMax(int ndim, const double* const a);

#endif
