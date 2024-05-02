//
//
//  Fisher vector, vlad and bovw representation
//
//  Created by Lázaro Janier Glez-Soler on 12/11/17.
//  Copyright © 2017 w. All rights reserved.
//

#include <fstream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>

//#include "common.h"
#include "pca.h"
#include "PDA_training.h"
#include "PDA_testing.h"
#include <dirent.h>
#include <getopt.h>

extern "C"
{
#include <vl/svm.h>
}

using namespace std;
using namespace cv;
using namespace dimension_reduction;

#define BUFFER_SIZE 1000

static void help()
{
	cout << "\nUtility app for presentation attack detection. \n" << endl;
	cout << "Created by BsC. Lazaro Janier Gonzalez Soler. \n" << endl;

	cout << "USAGE: \n"
		"\tParameters: \n"
		"\t[-m val]: algorithm mode (val=1: training, val=2: testing)\n"
		"\t[-i input_path]: path to input images\n"
		"\t[-o output_path]: path to output folder\n"
		"\t[-c num_centers]: number of center used by algorithms (default: 256)\n"
		"\t[-z num_features]: number of features used by pca, gmm or bovw training (default: 1000000)\n"
		"\t[-f scale_factor]: scale factor applied over whole input images (default: 1)\n"
		"EXAMPLE\n"
		"\t./PDAfV -m 1 -i ~/images_folder -o ~/output_folder -c 512 -z 1000000\n" << endl;

}

int LoadOptions(int argc, char **argv, extract_params* params, dsift_extraction_params* dsift_params)
{
	extern char * optarg;
	extern int optopt;
	int c;
	params->imageDir = (char *)malloc(sizeof(char)* 256);
	params->outputDir = (char *)malloc(sizeof(char)* 256);

	int it = 0;
	params->mode = 1;
	params->method = fv;
	dsift_params->windows_size = 1.5000;
	params->pca_dim = 64;
	params->sift_dim = 128;
	dsift_params->cSize = 4;
	dsift_params->magnif = 6;
	dsift_params->xy_step = 3;
	params->PCA = true;
	dsift_params->scale_factor = 1;
	params->num_cluster = 256;
	dsift_params->contrastthreshold = 0.0050000001;
	params->descriptor_size = 1000000;
	int binSz[4] = { 4, 6, 8, 10 };
	dsift_params->binSz = (int*)malloc(sizeof(int)*dsift_params->cSize);
	memcpy(dsift_params->binSz, binSz, sizeof(int)*dsift_params->cSize);

	while ((c = getopt(argc, argv, "a:m:n:i:o:s:f:c:z:p:r:")) != -1)
	{
		it++;
		switch (c)
		{
		case 'm':
			params->mode = atoi(optarg);
			break;
		case 'a':
			params->method = (encoding_method)atoi(optarg);
			break;
		case 'i':
			strcpy(params->imageDir, optarg);
			break;
		case 'o':
			strcpy(params->outputDir, optarg);
			break;
		case 's':
			dsift_params->xy_step = atoi(optarg);
			break;
		case 'f':
			dsift_params->scale_factor = (float)atof(optarg);
			break;
		case 'c':
			params->num_cluster = atoi(optarg);
			break;
		case 'e':
			break;
		case 'z':
			params->descriptor_size = atoi(optarg);
			break;
		case 'p':
			params->pca_dim = atoi(optarg);
			break;
		case 'n':
			params->database_name = optarg;
			break;
		case ':':
			printf("\n*** La opción -%c requiere de un parámetro\n", optopt);
			return 0;
		case '?':
			printf("\n*** Opción no reconocida: -%c\n", optopt);
			return 0;
		}
	}

	return 1;
}

void free_extract_params(extract_params* params, dsift_extraction_params* dsift_params)
{
	free(params->imageDir);
	free(params->outputDir);
	free(dsift_params->binSz);
}

int main(int argc, char * argv[])
{

	if (argc < 4)
	{
		help();
		return -1;
	}
	vector<string> image_path;
	/*LoadTextFile(argv[2], image_path);*/
	string output(argv[3]);
	extract_params params;
	struct dsift_extraction_params dsift_params;
	LoadOptions(argc, argv, &params, &dsift_params);

	switch (params.mode){
	case 1:
		///////////////////////////////////////////////////////
		// TRAIN ENCODER
		///////////////////////////////////////////////////////
		{
			  if (params.method == fv)
				  classifier_training::training_fv_encoder(params, dsift_params);
			  else if (params.method == vlad_m)
				  classifier_training::training_vlad_encoder(params, dsift_params);
			  else
				  classifier_training::training_bovw_encoder(params, dsift_params);
		}
		break;
	case 2:
		///////////////////////////////////////////////////////
		// TEST SAMPLES USING TRAINED DECODER
		///////////////////////////////////////////////////////
		{
			  if (params.method == fv)
				  classifier_testing::PDAfV_Classify(params, dsift_params);
			  else if (params.method == vlad_m)
				  classifier_testing::PDAVlad_Classify(params, dsift_params);
			  else
				  classifier_testing::PDABoW_Classify(params, dsift_params);
		}
		break;
	case 3:
		///////////////////////////////////////////////////////
		// EXTRACT AND SAVE SAMPLES DESCRIPTORS FOR GMM TRAINING
		///////////////////////////////////////////////////////
		//classifier_testing::PDAfV_Classify(image_path, output.c_str(), params, dsift_params);
		break;
	default:
		break;
	}

	free_extract_params(&params, &dsift_params);
	return 0;
}
