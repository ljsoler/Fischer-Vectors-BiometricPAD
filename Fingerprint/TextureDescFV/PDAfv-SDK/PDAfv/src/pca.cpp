/*
 * pca.cpp
 *
 *  Created on: Dec 5, 2011
 *      Author: kadir
 */

#include "pca.h"
#include <iostream>

//#include "TTrainedData.h"
//#include "TImagesHandle.h"
//#include "Janus_Common/Utils.h"

using namespace std;
using namespace cv;

namespace dimension_reduction{

	namespace pca{
								
		float* project(float* descriptors, int numOfData, int dimension, float* means, float* eigenvectors, int max_components)
		{
			for (int i = 0; i < numOfData; i++)
				for (int j = 0; j < dimension; j++)
					descriptors[i*dimension + j] -= means[j];
								
			//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, max_components, numOfData, dimension, 1.0, eigenvectors, dimension, descriptors, dimension, 0.0, result_temp, numOfData);


			Mat desc(numOfData, dimension, CV_32FC1, descriptors);
			Mat eigenVecT(max_components, dimension, CV_32FC1, eigenvectors);
			Mat desc_T;
			transpose(desc, desc_T);
			Mat output = eigenVecT*desc_T;
			Mat out;
			transpose(output, out);
			float* result = (float*)malloc(sizeof(float)*numOfData*max_components);
			for (int i = 0; i < numOfData; i++)
				for (int j = 0; j < max_components; j++)
					result[i*max_components + j] = out.at<float>(i, j);
			for (int i = 0; i < numOfData; i++)
				for (int j = 0; j < dimension; j++)
					descriptors[i*dimension + j] += means[j];
			return result;
		}

		void train_model(float* descriptors, int numOfData, int dimension, int max_components, bool whitening, float** means, float** eigenvectors, float** eigenvalues)
		{			
			float* descriptor_mean = (float*)malloc(sizeof(float)*dimension);
			//COMPUTING DESCRIPTOR MEANS
			for (int i = 0; i < dimension; i++)
			{
				descriptor_mean[i] = 0;
				for (int j = 0; j < numOfData; j++)
					descriptor_mean[i] += descriptors[j*dimension + i];
				descriptor_mean[i] /= (float)numOfData;
			}		
			*means = descriptor_mean;
			//SUBSTRACTING MEANS TO DESCRIPTORS
			float* x = (float*)malloc(sizeof(float)*numOfData*dimension);
			for (int i = 0; i < numOfData; i++)
			{
				for (int j = 0; j < dimension; j++)
					x[i*dimension + j] = descriptors[i*dimension + j] - descriptor_mean[j];				
			}
			//COMPUTING COVARIANCE MATRIX AND NORMALIZING
			cv::Mat xT_cov(numOfData, dimension, CV_32FC1, x);
			cv::Mat x_cov;
			transpose(xT_cov, x_cov);
			cv::Mat cov = x_cov*xT_cov;	
			float* cov_matrix = (float*)malloc(sizeof(float)*dimension*dimension);
			for (int i = 0; i < dimension; i++)
				for (int j = 0; j < dimension; j++)
				{
					float aux = (float)cov.at<float>(i,j);
					cov_matrix[i*dimension + j] = (float)aux/numOfData;
				}
			cv::Mat covariance(dimension, dimension, CV_32FC1, cov_matrix);			
			free(x);
			//free(x_T);
			cv::Mat E, V;			
			cv::eigen(covariance, E, V);
			free(cov_matrix);
			float* eigenVal = (float*)malloc(sizeof(float)*MIN(max_components, dimension)); 

			float* eigenV = (float*)malloc(sizeof(float)*MIN(max_components, dimension)*dimension);
			for (int i = 0; i < MIN(max_components, dimension); i++)
			{
				eigenVal[i] = (whitening)? E.at<float>(i, 0) + whiteningRegul * E.at<float>(0, 0): E.at<float>(i, 0);
				for (int j = 0; j < dimension; j++)
					eigenV[i*dimension + j] = (whitening)? (float)V.at<float>(i,j) / sqrtf(eigenVal[i]): V.at<float>(i,j);
			}
			*eigenvectors = eigenV;
			*eigenvalues = eigenVal;
		}

		int load_model(string path, float** means, float** eigenvectors, float** eigenvalues, int* max_components, int* dimension)
		{
			string t_path = path;
			size_t idx = path.find_last_of('.');
			t_path.insert(idx, "_info");
			FILE* f = fopen(t_path.c_str(), "rb");
			//STORE DESCRIPTOR SIZE
			int comp = 0;
			fscanf(f, "%d\n", &comp);	
			*max_components = comp;
			int dim = 0;
			fscanf(f, "%d\n", &dim);	
			*dimension = dim;
			fclose(f);
			
			f = fopen(path.c_str(), "rb");
			if (f == NULL)
				return -1;
			float* means_tmp = (float*)malloc(sizeof(float)*dim);
			float* eigenVal_tmp = (float*)malloc(sizeof(float)*comp);
			float* eigenVect_tmp = (float*)malloc(sizeof(float)*comp*dim);
			fread(means_tmp, sizeof(float), dim, f);
			fread(eigenVal_tmp, sizeof(float), comp, f);
			fread(eigenVect_tmp, sizeof(float), comp*dim, f);
			fclose(f);

			*means = means_tmp;
			*eigenvalues = eigenVal_tmp;
			*eigenvectors = eigenVect_tmp;

			return 1;
		}

		int save_model(string path, float* means, float* eigenvalues, float* eigenvectors, int max_components, int dimension)
		{
			FILE* f = fopen(path.c_str(), "wb");
			if (f == NULL)
				return -1;

			fwrite(means, sizeof(float), dimension, f);
			fwrite(eigenvalues, sizeof(float), max_components, f);
			fwrite(eigenvectors, sizeof(float), max_components*dimension, f);
	
			fclose(f);

			string t_path = path;
			size_t idx = path.find_last_of('.');
			t_path.insert(idx, "_info");
			f = fopen(t_path.c_str(), "w");
			//STORE DESCRIPTOR SIZE
			fprintf(f, "%d\n", max_components);	
			fprintf(f, "%d\n", dimension);	
			fclose(f);

			return 1;
		}
	}	
}
