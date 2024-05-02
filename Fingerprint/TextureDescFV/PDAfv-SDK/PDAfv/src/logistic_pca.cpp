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

#define LOGIT(X) log(X/(1.0 - X))
#define M 4.0

namespace dimension_reduction{

	namespace logistic_pca{

		double deviance(unsigned char* descriptors, float* descriptor_means, const Mat u, float* q, int numOfData, int dimension, int max_components)
		{
			float p = 0;
			for(int i = 0; i < dimension; i++)
					p += descriptor_means[i];

			Mat ut;
			transpose(u, ut);
			Mat cov = ut*u;

			float* r_mat = (float*)malloc(sizeof(float)*numOfData*dimension);
			for(int i = 0; i < numOfData*dimension; i++)
				r_mat[i] = M*q[i] - p;

			Mat r = Mat(numOfData, dimension, CV_32FC1, r_mat);
			Mat aux = r*cov + p;
			Mat x(numOfData, dimension, CV_8U, descriptors);
			Mat xt;
			Mat mu(1, dimension, CV_32FC1, descriptor_means);
//			Mat mut;
//			transpose(mu, mut);
			r += p;
			transpose(x, xt);
			double fst_term = (-2.0) * trace(xt*aux)[0];
			double snd_term = 0.0;
			for (int i = 0; i < numOfData; i++)
			{
				for (int j = 0; j < dimension; j++)
				{
					Mat uut = cov * (r.row(i) - mu);
					//double expterm = mu.row(j) + uut.row(j);
					//snd_term += log(1.0 + exp(expterm));
				}
			}
			snd_term *= 2.0;
			return fst_term + snd_term;
		}

		float* project(unsigned char* descriptors, int numOfData, int dimension, float* means, float* eigenvectors, int max_components)
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

		void train_model(unsigned char* descriptors, int numOfData, int dimension, int max_components, int max_iteration, float** means, float** eigenvectors, float** eigenvalues)
		{
			//CONVERTING BINARY DESCRIPTORS TO RANGE [-1, 1] -> Q
			float* q = (float*)malloc(sizeof(float)*numOfData*dimension);
#pragma omp parallel for
			for (int i = 0; i < numOfData; i++)
			{
				for (int j = 0; j < dimension; j++)
					q[i*dimension + j] = 2.0* descriptors[i*dimension + j] - 1.0;
			}

			float* descriptor_mean = (float*)malloc(sizeof(float)*dimension);
			float* eta = (float*)malloc(sizeof(float)*numOfData*dimension);
			//COMPUTING DESCRIPTOR MEANS
			for (int i = 0; i < dimension; i++)
			{
				float temp_mean = 0;
				for (int j = 0; j < numOfData; j++)
				{
					eta[j*dimension + i] = M*q[j*dimension + i];
					temp_mean += eta[j*dimension + i];
				}
				temp_mean /= (float)numOfData;
				descriptor_mean[i] = temp_mean;
			}		

			*means = descriptor_mean;
			//calculating covariance matrix qtq
			Mat q_mat = Mat(numOfData, dimension, CV_32FC1, q);
			Mat qt_mat;
			transpose(q_mat, qt_mat);
			Mat qtq = qt_mat*q_mat;
			//INITIALIZING U USING SVD DESCOMPOSITION
			Mat u_tmp, S, V;
			cv::SVDecomp(q_mat, S, u_tmp, V);
			Mat u_mat = Mat(V, Range::all(), Range(0, max_components));
			Mat ut_mat;
			transpose(u_mat, ut_mat);
			Mat uut = u_mat*ut_mat;

			//-----------COMPUTING THETA------------------
			//Substracting means to eta
			float* scale_eta = (float*)malloc(sizeof(float)*numOfData*dimension);
//#pragma omp parallel for
//			for (int i = 0; i < dimension; i++)
//				for (int j = 0; j < numOfData; j++)
//					scale_eta[j*dimension + i] =  [j*dimension + i] - descriptor_mean[i];

			Mat seta_mat = Mat(numOfData, dimension, CV_32FC1, scale_eta);
			Mat theta = seta_mat*uut;
			Mat mu = Mat(1, dimension, CV_32FC1, descriptor_mean);
			Mat first_term = Mat::ones(Size(numOfData, 1), CV_32FC1);
			theta += first_term;


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
