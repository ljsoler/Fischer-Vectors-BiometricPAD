// pda_demo.cpp : Defines the entry point for the console application.
//
//
//
//  Fisher vector, vlad and bovw representation
//
//  Created by Lázaro Janier Glez-Soler on 06/01/18.
//  Copyright © 2018 w. All rights reserved.
//

#include <fstream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>
#include "plbp.h"
#include "phog.h"
#include "pbsif.h"
#include "psurf.h"
#include "porb.h"
#include "pbrief.h"
#include "phow_dsift.h"
#include "pdafv_pipeline.h"
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

enum feature_extractor
{
	SIFT,
	LBP,
	HOG,
	BSIF, 
	SURF,
	ORBD,
	BRIEF
};

enum Generative_model
{
	GMM,
	BMM
};

static void help()
{
	cout << "\nSDK for presentation attack detection. \n" << endl;
	cout << "Created by MsC. Lazaro Janier Gonzalez Soler. \n" << endl;

	cout << "USAGE: \n"
		"\tParameters: \n"
		"\t[Cl]: Clustering algorithm selector (Cl=0: Using BMM, Cl=1: Using GMM)\n"
		"\t[-m val]: algorithm mode (val=1: training, val=2: testing)\n"
		"\t[-i input_path]: path to input images\n"
		"\t[-o output_path]: path to output folder\n"
		"\t[-c num_centers]: number of center used by algorithms (default: 256)\n"
		"\t[-d desciptor_to_encode]: texture descriptors will be encoded (SIFT=0, LBP = 1, HOG = 2, BSIF = 3, SURF = 4, ORB = 5, BRIEF = 6, default: SIFT)\n"
		"\t[-z num_features]: number of features used by pca, gmm or bovw training (default: 1000000)\n"
		"\t[-f scale_factor]: scale factor applied over whole input images (default: 1)\n"
		"\t[-n output_filename]: output name used for saving test results\n"
		"EXAMPLE\n"
		"training: \t./PDAfV 1 -m 1 -d 0 -i ~/images_folder -o ~/output_folder -c 512 -z 1000000\n"
		"testing: \t./PDAfV 1 -m 2 -d 0 -i ~/images_folder -o ~/output_folder -c 512 -n biometrika\n"<< endl;

}

int LoadOptions(int argc, char **argv, extract_params<float>* params)
{
	extern char * optarg;
	extern int optopt;
	int c;
	params->imageDir = (char *)malloc(sizeof(char)* 256);
	params->outputDir = (char *)malloc(sizeof(char)* 256);
	feature_extractor fe = SIFT;
	int it = 0;
	params->mode = 1;	
	params->pca_dim = 64;
	params->PCA_usage = true;
	params->features_dim = 128;		
	params->num_cluster = 256;
	params->threshold = 0;
	params->descriptors_count = 1000000;
	params->feature_extractor = compute_denseSIFT_for_image;

	while ((c = getopt(argc, argv, "d:m:n:i:o:c:z:p:t:")) != -1)
	{
		it++;
		switch (c)
		{
		case 'm':
			params->mode = atoi(optarg);
			break;
		case 'd':
			fe = (feature_extractor)atoi(optarg);
			switch (fe)
			{
			case SIFT:
				params->feature_extractor = compute_denseSIFT_for_image;				
				params->features_dim = 128;
				break;
			case LBP:
				params->feature_extractor = compute_lbp_for_image;
				params->features_dim = 256;
				params->PCA_usage = true;
				break;
			case HOG:
				params->feature_extractor = compute_phog_for_image;
				params->features_dim = 36;
				params->PCA_usage = false;
				break;	
			case BSIF:
				params->feature_extractor = compute_multiscale_bsif_for_image;
				params->features_dim = 128;
				break;
			case SURF:
				params->feature_extractor = compute_surf_for_image;
				params->features_dim = 64;
				break;
			case ORBD:
				params->feature_extractor = compute_orb_for_image_float;
				params->features_dim = 256;
				break;
			case BRIEF:
				params->feature_extractor = compute_brief_for_image_float;
				params->features_dim = 256;
				break;
			}
			break;
		case 'i':
			strcpy(params->imageDir, optarg);
			break;
		case 'o':
			strcpy(params->outputDir, optarg);
			break;		
		case 'c':
			params->num_cluster = atoi(optarg);
			break;
		case 't':
			params->threshold = atof(optarg);
			break;
		case 'e':
			break;
		case 'z':
			params->descriptors_count = atoi(optarg);
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

int LoadOptions(int argc, char **argv, extract_params<unsigned char>* params)
{
	extern char * optarg;
	extern int optopt;
	int c;
	params->imageDir = (char *)malloc(sizeof(char)* 256);
	params->outputDir = (char *)malloc(sizeof(char)* 256);
	feature_extractor fe = SIFT;
	int it = 0;
	params->mode = 1;
	params->pca_dim = 64;
	params->PCA_usage = false;
	params->features_dim = 128;
	params->num_cluster = 256;
	params->threshold = 0;
	params->descriptors_count = 1000000;
	params->feature_extractor = compute_orb_for_image;

	while ((c = getopt(argc, argv, "d:m:n:i:o:c:z:p:t:")) != -1)
	{
		it++;
		switch (c)
		{
		case 'm':
			params->mode = atoi(optarg);
			break;
		case 'd':
			fe = (feature_extractor)atoi(optarg);
			switch (fe)
			{
			case BSIF:
				params->feature_extractor = compute_multiscale_binary_bsif_for_image;
				params->features_dim = 8;
				break;
			case ORBD:
				params->feature_extractor = compute_orb_for_image;
				params->features_dim = 256;
				break;
			case BRIEF:
				params->feature_extractor = compute_brief_for_image;
				params->features_dim = 256;
				break;
			}
			break;
		case 'i':
			strcpy(params->imageDir, optarg);
			break;
		case 'o':
			strcpy(params->outputDir, optarg);
			break;
		case 'c':
			params->num_cluster = atoi(optarg);
			break;
		case 't':
			params->threshold = atof(optarg);
			break;
		case 'e':
			break;
		case 'z':
			params->descriptors_count = atoi(optarg);
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

template<class T>
void free_extract_params(extract_params<T>* params)
{
	free(params->imageDir);
	free(params->outputDir);	
}

int main(int argc, char * argv[])
{
	if (argc < 4)
	{
		help();
		return -1;
	}
	int mode = atoi(argv[1]);
	if(mode > 0) //ACTIVATED GMM OPTION
	{
		extract_params<float> params;
		LoadOptions(argc, argv, &params);
		switch(params.mode){
		case 1:
			///////////////////////////////////////////////////////
			// TRAIN ENCODER
			///////////////////////////////////////////////////////
			{
				pdafv::training_fv_encoder_gmm(params);
			}
			break;
		case 2:
			///////////////////////////////////////////////////////
			// TEST SAMPLES USING TRAINED DECODER
			///////////////////////////////////////////////////////
			{
				pdafv::testing_fv_encoder_gmm(params);
			}
			break;
		default:
			break;
		}
		free_extract_params(&params);
	}
	else //ACTIVATED BMM OPTION
	{
		extract_params<unsigned char> u_params;
		LoadOptions(argc, argv, &u_params);
		switch(u_params.mode){
		case 1:
			///////////////////////////////////////////////////////
			// TRAIN ENCODER
			///////////////////////////////////////////////////////
			{
				pdafv::training_fv_encoder_bmm(u_params);
			}
			break;
		case 2:
			///////////////////////////////////////////////////////
			// TEST SAMPLES USING TRAINED DECODER
			///////////////////////////////////////////////////////
			{
				pdafv::testing_fv_encoder_bmm(u_params);
			}
			break;
		default:
			break;
			free_extract_params(&u_params);
		}
	}

	return 0;
}


