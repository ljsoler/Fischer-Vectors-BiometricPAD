#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <string.h>
#include "math.h" 
#include <algorithm>
#include <limits>

using namespace std;
using namespace cv;

#define name_size 50

enum encoding_method
{
	bow,
	fv,
	vlad_m
};

struct extract_params
{
	encoding_method method;
	int mode;
	char* imageDir;
	char* outputDir;	
	int num_cluster;
	int sift_dim;	
	int pca_dim;	
	bool PCA;
	int descriptor_size;	
	string database_name;
};

struct dsift_extraction_params
{
	float scale_factor;
	int xy_step;
	int magnif;
	float windows_size;
	float contrastthreshold;
	int* binSz;
	int cSize;
};

int SaveDescrBinary(string path, const double* descr, long descr_size);

int SaveDescrBinary(string path, float* descr, long descr_size);

int LoadDescrBinary(string path, double* descr, long size);

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

#endif
