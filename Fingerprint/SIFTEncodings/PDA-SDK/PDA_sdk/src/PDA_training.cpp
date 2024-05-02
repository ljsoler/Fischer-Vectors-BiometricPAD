#include "PDA_training.h"
#include <stdio.h>
#include <math.h>

#include "kmeans.h"
#include "Vlad_encoding.h"
#include "Fisher_encoding.h"
#include "Bovw_encoding.h"
#include <vl/dsift.h>
#include <vector>
#include "common.h"
#include <sys/stat.h>
#include <vl/kmeans.h>
#include <vl/vlad.h>
#include <vl/fisher.h>
#include <vl/mathop.h>
#include <vl/svm.h>
#include "random_sampling.h"
#include <ctime>
#include <time.h>
#include <sys/timeb.h>
#ifdef WIN32
	#include <direct.h>
	#include <process.h>	
#else
#include <sys/types.h>
#include <unistd.h>
#endif

#define max_iter 100
#define epsilon 0.001
#define svm_C 10
#define nr_classes 2
#define sample_prefix "/sample_descriptors"
#define descriptors_prefix "/descriptors_chunk_"
#define descriptors_folder "/descriptors_folder"
#define fv_folder "/fv_folder_"
#define vlad_folder "/vlad_folder_"
#define bovw_folder "/bovw_folder_"
#define vlad_descriptor "/vlad_descriptors"
#define bovw_descriptor "/bovw_descriptors"
#define fv_descriptors_prefix "/fv_descriptors"
#define pca_descriptors_prefix "/pca_descriptors"
#define bovw_model_prefix "/bovw_model"
#define ext ".bin"
#define suffix "_info.bin"
#define svm_model_name "svm_model"
#define No_Image 40
#define bovw_dim 15360
#define MAX_FSIZE 8388608000

using namespace std;
using namespace cv;
using namespace dimension_reduction;
using namespace visual_features;
using namespace clustering_and_indexing;


/*--------------------------------------------------------------------------------------------------*/

namespace classifier_training
{
	void training_fv_encoder(extract_params params, dsift_extraction_params dsift_params)
	{
		/*-------------------------------obtain sample image descriptors-------------------------------*/
		vector<string> imgs_url;
		vector<string> img_bmp;
		get_files(params.imageDir, ".png", imgs_url);		
		get_files(params.imageDir, ".bmp", img_bmp);
		//get_files(params.imageDir, ".jpg", img_bmp);
		imgs_url.insert(imgs_url.end(), img_bmp.begin(), img_bmp.end());
		long descriptor_size = 0; 	
		int db_size = imgs_url.size();	
		float* samples_descriptors = NULL;	
		//timeb ini, end;


		////////////////////////////////////////////////////////////////////
		///		DSIFT EXTRACTION AND SAVING FOR EACH IMAGE				///
		///////////////////////////////////////////////////////////////////
	#ifdef WIN32
		mkdir((string(params.outputDir) + descriptors_folder).c_str());
	#else
		mkdir((string(params.outputDir) + descriptors_folder).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	#endif
	
		if(LoadDescrBinaryInfo(string(params.outputDir) + descriptors_folder + sample_prefix + suffix, &descriptor_size) > 0)
		{		
			printf("Loading dsift features...\n");
			samples_descriptors = (float*)malloc(sizeof(float)*descriptor_size);		
			LoadDescrBinary(string(params.outputDir) + descriptors_folder + sample_prefix + ext, samples_descriptors, descriptor_size);	
			string path_aux = string(params.outputDir) + "/db.info";
			loadDBInfo(path_aux, &db_size);
		}
		else
		{
			printf("Extracting and saving dsift features...\n");		
			//descriptor_size = 0;
			srand(getpid() + time(NULL));
			//random_shuffle(imgs_url.begin(), imgs_url.end());	
			//GetLabels(imgs_url, "/live/");
			samples_descriptors = (float*)malloc(sizeof(float)*params.descriptor_size*params.sift_dim);								
			int numDescrsPerImage = params.descriptor_size/db_size;
			descriptor_size = params.descriptor_size*params.sift_dim;
	#pragma omp parallel for
			for (int i = 0; i < db_size; i++)
			{				
				printf("SIFT extraction for image %d\n", i);
				Mat img_i = readImage(imgs_url[i]);
				if (img_i.empty()) {
					cout << "ERROR" << endl;
					continue;
				}	

				if(dsift_params.scale_factor > 1)
				{
					Mat temp;
					resize(img_i, temp, Size(img_i.cols*dsift_params.scale_factor, img_i.rows*dsift_params.scale_factor));
					img_i = temp;			
				}
				//printf("Proccesing image %i from %i\n", i + 1, imgs_url.size());
				//ftime(&ini);
				vector<float> desc = compute_for_image(img_i, dsift_params);
				//ftime(&end);

				//printf("Time %d\n", (end.time - ini.time) * 1000 + (end.millitm - ini.millitm));

				int keypoint_size = desc.size()/params.sift_dim;
				//printf("Keypoint %i\n", keypoint_size);
				//SELECTING RANDOM VECTORS FROM EACH IMAGE FOR MODEL TRAINING
				int* index = (int*)malloc(sizeof(int)*keypoint_size);
				for (int j = 0; j < keypoint_size; j++)
					index[j] = j;
				shuffle(index, keypoint_size);
				for (int j = 0; j < numDescrsPerImage; j++)
				{				
					for (int k = 0; k < params.sift_dim; k++)
						samples_descriptors[i*numDescrsPerImage*params.sift_dim + j*params.sift_dim + k] = desc[index[j] * params.sift_dim + k];
				}			
				free(index);
			}

//			string aux = string(params.outputDir) + descriptors_folder + sample_prefix + "txt";
//			FILE* fl = fopen(aux.c_str(), "wb");
//			for (int j = 0; j < descriptor_size; j++)
//			{
//				if(j != 0 && j % 127 == 0)
//					fprintf(fl, "%f\n", samples_descriptors[j]);
//				else
//					fprintf(fl, "%f,", samples_descriptors[j]);
//			}
//			fclose(fl);
			SaveDescrBinary(string(params.outputDir) + descriptors_folder + sample_prefix + ext, samples_descriptors, descriptor_size);
			string path_aux = string(params.outputDir) + "/db.info";
			SaveDBInfo(path_aux, db_size);
		}
	
		/*---------------------------------------------------------------------------------------------*/
	
		/*----------------------------------learn PCA projection---------------------------------------*/	
		////////////////////////////////////////////////////////////////////
		///		DIMENSION REDUCTION OF SELECTED FEATURES                ///
		///////////////////////////////////////////////////////////////////
		float* descriptors_means = NULL;
		float* eigenvalues = NULL;
		float* eigenvectors = NULL;
		int dim = 0;
		int max_components = 0;
		if(params.PCA)
		{	
			string pca_path(params.outputDir);
			pca_path += "/pca_model.bin";	
			if(!is_file_exist(pca_path.c_str()))		
			{			
				printf("Training PCA model...\n");												
				pca::train_model(samples_descriptors, descriptor_size/params.sift_dim, params.sift_dim, params.pca_dim, false, &descriptors_means, &eigenvectors, &eigenvalues);
				pca::save_model(pca_path, descriptors_means, eigenvalues, eigenvectors, params.pca_dim, params.sift_dim);	
				dim = params.sift_dim;
				max_components = params.pca_dim;
			}
			else
				pca::load_model(pca_path, &descriptors_means, &eigenvectors, &eigenvalues, &max_components, &dim);		
		}

		/*---------------------------------------------------------------------------------------------*/	
		///////////////////////////////////////////////
		///		TRAINING GMM MODELS                ///
		//////////////////////////////////////////////		
		float* means, *sigmas, *weights = NULL;		
		bool mark = false;
		string output_path(params.outputDir);
		if (gmm::load_gmm_results_binary(output_path + "/", "pad", means, sigmas, weights, params.num_cluster, params.pca_dim) < 0)
		{
			printf("Training GMM model...\n");
			//Projecting DSIFT descriptor with PCA model
			int dimension = params.sift_dim;
			int total_data = 0;
			if(params.PCA)
			{				
				float* reduced_descriptors = pca::project(samples_descriptors, descriptor_size/dim, dim, descriptors_means, eigenvectors, max_components);

				free(samples_descriptors);
				samples_descriptors = reduced_descriptors;
				descriptor_size = (descriptor_size/dim)*max_components;				
				dimension = max_components;
				total_data = descriptor_size/max_components;
//				string aux = string(params.outputDir) + descriptors_folder + "/pca_samples_descriptors.csv";
//				FILE* fl = fopen(aux.c_str(), "wb");
//				for (int j = 0; j < total_data*params.pca_dim; j++)
//				{
//					if(j != 0 && (j + 1) % params.pca_dim == 0)
//						fprintf(fl, "%f\n", samples_descriptors[j]);
//					else
//						fprintf(fl, "%f,", samples_descriptors[j]);
//				}
//				fclose(fl);
				/*string output_reduced(params.outputDir);
				output_reduced += "/reduced_descirptor.bin";
				SaveDescrBinary(output_reduced, samples_descriptors, descriptor_size);*/
				//memcpy(samples_descriptors, &reduced_descriptors.data[0], sizeof(float)*descriptor_size);
			}
			//Training GMM model
			VlGMM* gmm_model = gmm::train_gmm_model(samples_descriptors, total_data, dimension, params.num_cluster, max_iter, max_iter);
			printf("\tSave gmm model...\n");
			gmm::save_gmm_results_binary(params.outputDir, "pad", gmm_model);
			float* means_tmp = (float*)vl_gmm_get_means(gmm_model);
			float* sigmas_tmp = (float*)vl_gmm_get_covariances(gmm_model);
			float* weights_tmp = (float*)vl_gmm_get_priors(gmm_model);
			means = (float*)malloc(sizeof(float)*params.num_cluster*dimension);
			sigmas = (float*)malloc(sizeof(float)*params.num_cluster*dimension);
			weights = (float*)malloc(sizeof(float)*params.num_cluster);
			memcpy(means, means_tmp, sizeof(float)*params.num_cluster*dimension);
			memcpy(sigmas, sigmas_tmp, sizeof(float)*params.num_cluster*dimension);
			memcpy(weights, weights_tmp, sizeof(float)*params.num_cluster);
			vl_gmm_delete(gmm_model);
		}
		free(samples_descriptors);
		descriptor_size = 0;
		////////////////////////////////////////////////////////////////////
		///		FISHER ENCODING FOR EACH DATABASE IMAGE                 ///
		///////////////////////////////////////////////////////////////////
		printf("Loading features for fv encoding...\n");
#ifdef WIN32
		mkdir((string(params.outputDir) + fv_folder + to_string(params.num_cluster)).c_str());
#else
		mkdir((string(params.outputDir) + fv_folder + to_string(params.num_cluster)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
		
		string path_aux = string(params.outputDir) + "/db.info";		
		//loadDBInfo(path_aux, &db_size);
		db_size = imgs_url.size();
		//Proccesing chunks for image encoding		
		string tmp_fv = string(params.outputDir) + fv_folder + to_string(params.num_cluster) + fv_descriptors_prefix + ext;
		string t_path = tmp_fv;
		size_t idx = tmp_fv.find_last_of('.');						
		t_path.insert(idx, "_info");
		if(!is_file_exist(t_path.c_str()) && !is_file_exist(tmp_fv.c_str()))
		{				
			float* fv_descriptors = (float*)malloc(sizeof(float)*db_size*2*params.num_cluster*params.pca_dim);
			dim = 2 * params.pca_dim* params.num_cluster;
#pragma omp parallel for
			for (int i = 0; i < db_size; i++)
			{					
				printf("FV encoding image %d\n", i);
				//Load offset for each image inside chunk
				Mat img_i = readImage(imgs_url[i]);
				if (img_i.empty()) {
					cout << "ERROR" << endl;
					continue;
				}	
				
				if (dsift_params.scale_factor > 1)
				{
					Mat temp;
					resize(img_i, temp, Size(img_i.cols*dsift_params.scale_factor, img_i.rows*dsift_params.scale_factor));
					img_i = temp;
				}
				vector<float> desc = compute_for_image(img_i, dsift_params);								

				//DSIFT dimension reduction 								
				float* feat_proj = pca::project(&desc[0], desc.size()/params.sift_dim, params.sift_dim, descriptors_means, eigenvectors, max_components);
				long tmp_size = (desc.size()/params.sift_dim)*max_components;				

				float* enc = fisher::fisher_encoding(means, sigmas, weights, feat_proj, tmp_size/max_components, max_components, params.num_cluster);
				free(feat_proj);
				for (int j = 0; j < dim; j++)
					fv_descriptors[i*dim + j] = enc[j];
				//memcpy(&(fv_descriptors[fv_offset]), enc, sizeof(float)*2*params.num_cluster*params.pca_dim);
				//free(img_descriptor);		
				vl_free(enc);								
			}			
			SaveDescrBinary(tmp_fv, fv_descriptors, db_size*2*params.num_cluster*params.pca_dim);
			free(fv_descriptors);			
		}		
		if(params.PCA)
		{
			free(descriptors_means);
			free(eigenvalues);
			free(eigenvectors);
		}
		free(sigmas);
		free(means);
		free(weights);
		/////////////////////////////////////////////////
		///		         TRAINING SVM		          ///
		////////////////////////////////////////////////

		string model_path(string(params.outputDir) + '/' + svm_model_name + ext);	
		if(!is_file_exist(model_path.c_str()))
		{
			printf("Training svm model...\n");		
			string tmp_fv = string(params.outputDir) + fv_folder + to_string(params.num_cluster) + fv_descriptors_prefix + ext;
			string t_path = tmp_fv;
			size_t idx = tmp_fv.find_last_of('.');						
			t_path.insert(idx, "_info");
			descriptor_size = 0;
			dim = 2*params.num_cluster*params.pca_dim;
			if(LoadDescrBinaryInfo(t_path, &descriptor_size) < 0)
			{
				cout << "ERROR LOADING FV DESCRIPTORS";
				return;
			}		
			db_size = imgs_url.size();
			float* fv_descriptors = (float*)malloc(descriptor_size*sizeof(float));			
			LoadDescrBinary(tmp_fv, fv_descriptors, descriptor_size);		
		
			int* index = (int*)malloc(sizeof(int)*db_size);
			for (int i = 0; i < db_size; i++)
				index[i] = i;
			float* model[nr_classes];
			float bias[nr_classes];
			int model_dim = 0;		

			for (int i = 0; i < nr_classes; i++)
			{
				const char* class_name = i == 0? "fake": "live";
				printf("Training svm for %s class\n", class_name);			
				shuffle(index, db_size);
				double* labels = GetLabels(imgs_url, class_name);
				double* random_labels = (double*)malloc(sizeof(double)*db_size);
				float* random_fv = (float*)malloc(descriptor_size*sizeof(float));
				for (int j = 0; j < db_size; j++)
				{
					memcpy(&(random_fv[j*dim]), &(fv_descriptors[index[j] * dim]), sizeof(float)*dim);
					random_labels[j] = labels[index[j]];
				}								

				//Setting svm paramters
				VlSvmDataset* dataset = vl_svmdataset_new(VL_TYPE_FLOAT, random_fv, dim, db_size);
				double lambda = (double)1/(svm_C*db_size);
				
				VlSvm* svm = vl_svm_new_with_dataset(VlSvmSolverSdca, dataset, random_labels, lambda);
				vl_svm_set_bias_multiplier(svm, 1);
				vl_svm_set_loss(svm, VlSvmLossHinge);
				vl_svm_set_epsilon(svm, epsilon);
				vl_svm_set_max_num_iterations(svm, max_iter*db_size);

				vl_svm_train(svm);

				const double* m = vl_svm_get_model(svm);
				bias[i] = vl_svm_get_bias(svm);
				model_dim = vl_svm_get_dimension(svm);
				const VlSvmStatistics* statistics = vl_svm_get_statistics(svm);
				printf("loss = %f and dual_loss = %f\n", statistics->loss, statistics->dualLoss);
				switch (statistics->status)
				{
				case VlSvmStatusConverged:
						printf("Solver converged...\n");
						break;
				case VlSvmStatusMaxNumIterationsReached:
					printf("Iteration number reached...\n");
					break;			
				}

				model[i] = (float*)malloc(sizeof(float)*model_dim);

				for (int j = 0; j < model_dim; j++)
					model[i][j] = m[j];
				
				vl_svmdataset_delete(dataset);
				vl_svm_delete(svm);
				free(random_fv);
				free(random_labels);
				free(labels);
			}
			free(fv_descriptors);
			free(index);
			save_model(model_path, model, bias, model_dim, nr_classes);
			for(int i = 0; i < nr_classes; i++)
				free(model[i]);
		}
	}	

	void training_vlad_encoder(extract_params params, dsift_extraction_params dsift_params)
	{
		/*-------------------------------obtain sample image descriptors-------------------------------*/
		vector<string> imgs_url;
		vector<string> img_bmp;
		get_files(params.imageDir, ".png", imgs_url);		
		get_files(params.imageDir, ".bmp", img_bmp);
		imgs_url.insert(imgs_url.end(), img_bmp.begin(), img_bmp.end());
		long descriptor_size = 0; 	
		int db_size = imgs_url.size();	
		float* samples_descriptors = NULL;	
		////////////////////////////////////////////////////////////////////
		///		DSIFT EXTRACTION AND SAVING FOR EACH IMAGE				///
		///////////////////////////////////////////////////////////////////
	#ifdef WIN32
		mkdir(string(params.outputDir).c_str());
		mkdir((string(params.outputDir) + descriptors_folder).c_str());
	#else
		mkdir(string(params.outputDir).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		mkdir((string(params.outputDir) + descriptors_folder).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	#endif
	
		if(LoadDescrBinaryInfo(string(params.outputDir) + descriptors_folder + sample_prefix + suffix, &descriptor_size) > 0)
		{		
			printf("Loading dsift features...\n");
			samples_descriptors = (float*)malloc(sizeof(float)*descriptor_size);		
			LoadDescrBinary(string(params.outputDir) + descriptors_folder + sample_prefix + ext, samples_descriptors, descriptor_size);	
			string path_aux = string(params.outputDir) + "/db.info";
			loadDBInfo(path_aux, &db_size);
		}
		else
		{
			printf("Extracting and saving dsift features...\n");		
			//descriptor_size = 0;
			srand(getpid() + time(NULL));
			//random_shuffle(imgs_url.begin(), imgs_url.end());	
			//GetLabels(imgs_url, "/live/");
			samples_descriptors = (float*)malloc(sizeof(float)*params.descriptor_size*params.sift_dim);								
			int numDescrsPerImage = params.descriptor_size/db_size;
			descriptor_size = params.descriptor_size*params.sift_dim;
	#pragma omp parallel for
			for (int i = 0; i < db_size; i++)
			{				
				Mat img_i = readImage(imgs_url[i]);
				if (img_i.empty()) {
					cout << "ERROR" << endl;
					continue;
				}	

				if(dsift_params.scale_factor > 1)
				{
					Mat temp;
					resize(img_i, temp, Size(img_i.cols*dsift_params.scale_factor, img_i.rows*dsift_params.scale_factor));
					img_i = temp;			
				}
				//printf("Proccesing image %i from %i\n", i + 1, imgs_url.size());					
				vector<float> desc = compute_for_image(img_i, dsift_params);				

				int keypoint_size = desc.size()/params.sift_dim;								
				//SELECTING RANDOM VECTORS FROM EACH IMAGE FOR MODEL TRAINING 
				int* index = (int*)malloc(sizeof(int)*keypoint_size);
				for (int j = 0; j < keypoint_size; j++)
					index[j] = j;
				//shuffle(index, keypoint_size);
				for (int j = 0; j < numDescrsPerImage; j++)
				{				
					for (int k = 0; k < params.sift_dim; k++)
						samples_descriptors[i*numDescrsPerImage*params.sift_dim + j*params.sift_dim + k] = desc[index[j] * params.sift_dim + k];
				}			
				free(index);			
			}
			SaveDescrBinary(string(params.outputDir) + descriptors_folder + sample_prefix + ext, samples_descriptors, descriptor_size);
			string path_aux = string(params.outputDir) + "/db.info";
			SaveDBInfo(path_aux, db_size);
		}
	
		/*---------------------------------------------------------------------------------------------*/
	
		/*----------------------------------learn PCA projection---------------------------------------*/	
		////////////////////////////////////////////////////////////////////
		///		DIMENSION REDUCTION OF SELECTED FEATURES                ///
		///////////////////////////////////////////////////////////////////
		float* descriptors_means = NULL;
		float* eigenvalues = NULL;
		float* eigenvectors = NULL;
		int dim = params.sift_dim;		
		int total_data = descriptor_size/dim;
		if (params.PCA)
		{
			int max_components = params.pca_dim;
			string pca_path(params.outputDir);
			pca_path += "/pca_model.bin";
			if (!is_file_exist(pca_path.c_str()))
			{
				printf("Training PCA model...\n");
				pca::train_model(samples_descriptors, descriptor_size / params.sift_dim, params.sift_dim, params.pca_dim, false, &descriptors_means, &eigenvectors, &eigenvalues);
				pca::save_model(pca_path, descriptors_means, eigenvalues, eigenvectors, params.pca_dim, params.sift_dim);
				dim = params.sift_dim;
				max_components = params.pca_dim;
			}
			else
				pca::load_model(pca_path, &descriptors_means, &eigenvectors, &eigenvalues, &max_components, &dim);

			float* reduced_descriptors = pca::project(samples_descriptors, descriptor_size / dim, dim, descriptors_means, eigenvectors, max_components);
			free(samples_descriptors);
			samples_descriptors = reduced_descriptors;
			descriptor_size = (descriptor_size / dim)*max_components;
			dim = max_components;
			total_data = descriptor_size / max_components;
		}		

		/*---------------------------------------------------------------------------------------------*/	
		////////////////////////////////////////////////////////////////////
		///		TRAINING VISUAL VOCABULARY				                ///
		///////////////////////////////////////////////////////////////////		
		printf("Building VQ model...\n");
		string bovw_path(params.outputDir);
		bovw_path += bovw_model_prefix;
		bovw_path += ext;
		float* centers;
		int num_center = params.num_cluster;				
		if(kmeans::load_kmeans_results_binary(bovw_path, &centers, dim, num_center) < 0)
		{
			VlKMeans* kmeans = kmeans::train_kmeans_model(samples_descriptors, total_data, dim, params.num_cluster);
			float* centers_tmp = (float*)vl_kmeans_get_centers(kmeans);	
			centers = (float*)malloc(sizeof(float)*vl_kmeans_get_num_centers(kmeans)*vl_kmeans_get_dimension(kmeans));
			memcpy(centers, centers_tmp, sizeof(float)*vl_kmeans_get_num_centers(kmeans)*vl_kmeans_get_dimension(kmeans));
			kmeans::save_kmeans_results_binary(bovw_path, kmeans);
			vl_kmeans_delete(kmeans);
		}		
		
		VlKDForest* forest = kmeans::build_kdtree(centers, num_center, dim, 2);

		//string path_aux = string(params.outputDir) + "/db.info";		
		//loadDBInfo(path_aux, &db_size);
		db_size = imgs_url.size();
		//Proccesing chunks for image encoding	
		#ifdef WIN32
		mkdir((string(params.outputDir) + vlad_folder + to_string(params.num_cluster)).c_str());
#else
		mkdir((string(params.outputDir) + vlad_folder + to_string(params.num_cluster)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
		string tmp_bovw = string(params.outputDir) + vlad_folder + to_string(params.num_cluster) + vlad_descriptor + ext;
		string t_path = tmp_bovw;
		size_t idx = tmp_bovw.find_last_of('.');						
		t_path.insert(idx, "_info");
		if(!is_file_exist(t_path.c_str()) && !is_file_exist(tmp_bovw.c_str()))
		{		

			float* vlad_descriptors = (float*)malloc(sizeof(float)*db_size*num_center*dim);	
			int vlad_dim = num_center*dim;
#pragma omp parallel for 
			for (int i = 0; i < db_size; i++)
			{					
				printf("Vlad encoding image %d\n", i);
				//Load offset for each image inside chunk
				Mat img_i = readImage(imgs_url[i]);
				if (img_i.empty()) {
					cout << "ERROR" << endl;
					continue;
				}	
				
				if(dsift_params.scale_factor > 1)
				{
					Mat temp;
					resize(img_i, temp, Size(img_i.cols*dsift_params.scale_factor, img_i.rows*dsift_params.scale_factor));
					img_i = temp;
				}

				vector<float> desc = compute_for_image(img_i, dsift_params);								

				//DSIFT dimension reduction 								
				float* feat_proj = pca::project(&desc[0], desc.size()/params.sift_dim, params.sift_dim, descriptors_means, eigenvectors, dim);
				long tmp_size = (desc.size()/params.sift_dim)*dim;				

				float* enc = vlad::vlad_encoding(forest, centers, feat_proj, tmp_size/ dim, dim, num_center);
				free(feat_proj);
				for (int j = 0; j < vlad_dim; j++)
					vlad_descriptors[i*vlad_dim + j] = enc[j];
				//memcpy(&(fv_descriptors[fv_offset]), enc, sizeof(float)*2*params.num_cluster*params.pca_dim);
				
				////free(img_descriptor);		
				vl_free(enc);								
			}			
			SaveDescrBinary(tmp_bovw, vlad_descriptors, db_size*num_center*dim);
			free(vlad_descriptors);			
		}
		free(centers);
		vl_kdforest_delete(forest);

		if(params.PCA)
		{
			free(descriptors_means);
			free(eigenvalues);
			free(eigenvectors);
		}
		/////////////////////////////////////////////////
		///		         TRAINING SVM		          ///
		////////////////////////////////////////////////

		string model_path(string(params.outputDir) + '/' + svm_model_name + ext);
		if (!is_file_exist(model_path.c_str()))
		{
			printf("Training svm model...\n");
			string tmp_fv = string(params.outputDir) + vlad_folder + to_string(params.num_cluster) + vlad_descriptor + ext;
			string t_path = tmp_fv;
			size_t idx = tmp_fv.find_last_of('.');
			t_path.insert(idx, "_info");
			descriptor_size = 0;
			dim = params.num_cluster*params.pca_dim;
			if (LoadDescrBinaryInfo(t_path, &descriptor_size) < 0)
			{
				cout << "ERROR LOADING FV DESCRIPTORS";
				return;
			}
			db_size = imgs_url.size();
			float* fv_descriptors = (float*)malloc(descriptor_size*sizeof(float));
			LoadDescrBinary(tmp_fv, fv_descriptors, descriptor_size);

			int* index = (int*)malloc(sizeof(int)*db_size);
			for (int i = 0; i < db_size; i++)
				index[i] = i;
			float* model[nr_classes];
			float bias[nr_classes];
			int model_dim = 0;

			for (int i = 0; i < nr_classes; i++)
			{
				const char* class_name = i == 0 ? "fake" : "live";
				printf("Training svm for %s class\n", class_name);
				shuffle(index, db_size);
				double* labels = GetLabels(imgs_url, class_name);
				double* random_labels = (double*)malloc(sizeof(double)*db_size);
				float* random_fv = (float*)malloc(descriptor_size*sizeof(float));
				for (int j = 0; j < db_size; j++)
				{
					memcpy(&(random_fv[j*dim]), &(fv_descriptors[index[j] * dim]), sizeof(float)*dim);
					random_labels[j] = labels[index[j]];
				}

				//Setting svm paramters
				VlSvmDataset* dataset = vl_svmdataset_new(VL_TYPE_FLOAT, random_fv, dim, db_size);
				double lambda = (double)1 / (svm_C*db_size);
				VlSvm* svm = vl_svm_new_with_dataset(VlSvmSolverSdca, dataset, random_labels, lambda);
				vl_svm_set_bias_multiplier(svm, 1);
				vl_svm_set_loss(svm, VlSvmLossHinge);
				vl_svm_set_epsilon(svm, epsilon);
				vl_svm_set_max_num_iterations(svm, max_iter*db_size);

				vl_svm_train(svm);

				const double* m = vl_svm_get_model(svm);
				bias[i] = vl_svm_get_bias(svm);
				model_dim = vl_svm_get_dimension(svm);
				const VlSvmStatistics* statistics = vl_svm_get_statistics(svm);
				printf("loss = %f and dual_loss = %f\n", statistics->loss, statistics->dualLoss);
				switch (statistics->status)
				{
				case VlSvmStatusConverged:
					printf("Solver converged...\n");
					break;
				case VlSvmStatusMaxNumIterationsReached:
					printf("Iteration number reached...\n");
					break;
				}

				model[i] = (float*)malloc(sizeof(float)*model_dim);

				for (int j = 0; j < model_dim; j++)
					model[i][j] = m[j];

				vl_svmdataset_delete(dataset);
				vl_svm_delete(svm);
				free(random_fv);
				free(random_labels);
				free(labels);
			}
			free(fv_descriptors);
			free(index);
			save_model(model_path, model, bias, model_dim, nr_classes);
			for (int i = 0; i < nr_classes; i++)
				free(model[i]);
		}		
	}	

	void training_bovw_encoder(extract_params params, dsift_extraction_params dsift_params)
	{
		/*-------------------------------obtain sample image descriptors-------------------------------*/
		vector<string> imgs_url;
		vector<string> img_bmp;
		vector<string> img_jpg;
		get_files(params.imageDir, ".png", imgs_url);
		get_files(params.imageDir, ".jpg", img_jpg);
		get_files(params.imageDir, ".bmp", img_bmp);
		imgs_url.insert(imgs_url.end(), img_bmp.begin(), img_bmp.end());
		imgs_url.insert(imgs_url.end(), img_jpg.begin(), img_jpg.end());
		long descriptor_size = 0; 	
		int db_size = imgs_url.size();	
		float* samples_descriptors = NULL;	
		////////////////////////////////////////////////////////////////////
		///		DSIFT EXTRACTION AND SAVING FOR EACH IMAGE				///
		///////////////////////////////////////////////////////////////////
	#ifdef WIN32
		mkdir(string(params.outputDir).c_str());
		mkdir((string(params.outputDir) + descriptors_folder).c_str());
	#else
		mkdir(string(params.outputDir).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		mkdir((string(params.outputDir) + descriptors_folder).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	#endif
	
		if(LoadDescrBinaryInfo(string(params.outputDir) + descriptors_folder + sample_prefix + suffix, &descriptor_size) > 0)
		{		
			printf("Loading dsift features...\n");
			samples_descriptors = (float*)malloc(sizeof(float)*descriptor_size);		
			LoadDescrBinary(string(params.outputDir) + descriptors_folder + sample_prefix + ext, samples_descriptors, descriptor_size);	
			string path_aux = string(params.outputDir) + "/db.info";
			loadDBInfo(path_aux, &db_size);
		}
		else
		{
			printf("Extracting and saving dsift features...\n");		
			//descriptor_size = 0;
			srand(getpid() + time(NULL));
			random_shuffle(imgs_url.begin(), imgs_url.end());	
			//GetLabels(imgs_url, "/live/");
			samples_descriptors = (float*)malloc(sizeof(float)*params.descriptor_size*params.sift_dim);								
			int numDescrsPerImage = params.descriptor_size/No_Image;
			descriptor_size = params.descriptor_size*params.sift_dim;
	#pragma omp parallel for
			for (int i = 0; i < No_Image; i++)
			{				
				Mat img_i = readImage(imgs_url[i]);
				if (img_i.empty()) {
					cout << "ERROR" << endl;
					continue;
				}	

				if(dsift_params.scale_factor > 1)
				{
					Mat temp;
					resize(img_i, temp, Size(img_i.cols*dsift_params.scale_factor, img_i.rows*dsift_params.scale_factor));
					img_i = temp;			
				}
				printf("Proccesing image %i\n", i + 1);					
				vector<float> desc = compute_for_image(img_i, dsift_params);				

				int keypoint_size = desc.size()/params.sift_dim;								
				//SELECTING RANDOM VECTORS FROM EACH IMAGE FOR MODEL TRAINING 
				int* index = (int*)malloc(sizeof(int)*keypoint_size);
				for (int j = 0; j < keypoint_size; j++)
					index[j] = j;
				shuffle(index, keypoint_size);
				for (int j = 0; j < numDescrsPerImage; j++)
				{				
					for (int k = 0; k < params.sift_dim; k++)
						samples_descriptors[i*numDescrsPerImage*params.sift_dim + j*params.sift_dim + k] = desc[index[j] * params.sift_dim + k];
				}			
				free(index);			
			}					
			SaveDescrBinary(string(params.outputDir) + descriptors_folder + sample_prefix + ext, samples_descriptors, descriptor_size);
			string path_aux = string(params.outputDir) + "/db.info";
			SaveDBInfo(path_aux, No_Image);
		}
	
		/*---------------------------------------------------------------------------------------------*/	
		int dim = params.sift_dim;		
		int total_data = descriptor_size/dim;
		/*---------------------------------------------------------------------------------------------*/	
		////////////////////////////////////////////////////////////////////
		///		TRAINING VISUAL VOCABULARY				                ///
		///////////////////////////////////////////////////////////////////		
		printf("Building VQ model...\n");
		string bovw_path(params.outputDir);
		bovw_path += bovw_model_prefix;
		bovw_path += ext;
		float* centers;
		int num_center = params.num_cluster;				
		if(kmeans::load_kmeans_results_binary(bovw_path, &centers, dim, num_center) < 0)
		{
			VlKMeans* kmeans = kmeans::train_kmeans_model(samples_descriptors, total_data, dim, params.num_cluster);
			float* centers_tmp = (float*)vl_kmeans_get_centers(kmeans);	
			centers = (float*)malloc(sizeof(float)*vl_kmeans_get_num_centers(kmeans)*vl_kmeans_get_dimension(kmeans));
			memcpy(centers, centers_tmp, sizeof(float)*vl_kmeans_get_num_centers(kmeans)*vl_kmeans_get_dimension(kmeans));
			kmeans::save_kmeans_results_binary(bovw_path, kmeans);
			vl_kmeans_delete(kmeans);
		}		
		
		VlKDForest* forest = kmeans::build_kdtree(centers, num_center, dim, 2);

		//string path_aux = string(params.outputDir) + "/db.info";		
		//loadDBInfo(path_aux, &db_size);
		db_size = imgs_url.size();
		//Proccesing chunks for image encoding	
		#ifdef WIN32
		mkdir((string(params.outputDir) + bovw_folder + to_string(params.num_cluster)).c_str());
#else
		mkdir((string(params.outputDir) + bovw_folder + to_string(params.num_cluster)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
		string tmp_bovw = string(params.outputDir) + bovw_folder + to_string(params.num_cluster) + bovw_descriptor + ext;
		string t_path = tmp_bovw;
		int idx = tmp_bovw.find_last_of('.');						
		t_path.insert(idx, "_info");
		if(!is_file_exist(t_path.c_str()) && !is_file_exist(tmp_bovw.c_str()))
		{		
			float* bovw_descriptors = (float*)malloc(sizeof(float)*db_size*bovw_dim);				
#pragma omp parallel for 
			for (int i = 0; i < db_size; i++)
			{					
				printf("BoW encoding image %d\n", i);
				//Load offset for each image inside chunk
				Mat img_i = readImage(imgs_url[i]);
				if (img_i.empty()) {
					cout << "ERROR" << endl;
					continue;
				}	
				
				if(dsift_params.scale_factor > 1)
				{
					Mat temp;
					resize(img_i, temp, Size(img_i.cols*dsift_params.scale_factor, img_i.rows*dsift_params.scale_factor));
					img_i = temp;
				}

				vector<float> frames_denseSift;
				vector<float> desc = compute_for_image(img_i, dsift_params, frames_denseSift);								

				//DSIFT dimension reduction 								
				float* feat_proj = &desc[0];
				long tmp_size = (desc.size()/params.sift_dim)*dim;				

				float* enc = bovw::bovw_encoding(forest, centers, feat_proj, tmp_size/ dim, dim, num_center, img_i.cols, img_i.rows, &frames_denseSift[0]);

				for (int j = 0; j < bovw_dim; j++)
					bovw_descriptors[i*bovw_dim + j] = enc[j];
					
				vl_free(enc);								
			}			
			SaveDescrBinary(tmp_bovw, bovw_descriptors, db_size*bovw_dim);
			free(bovw_descriptors);			
		}
		free(centers);
		vl_kdforest_delete(forest);
		/////////////////////////////////////////////////
		///		         TRAINING SVM		          ///
		////////////////////////////////////////////////

		string model_path(string(params.outputDir) + '/' + svm_model_name + ext);
		if (!is_file_exist(model_path.c_str()))
		{
			printf("Training svm model...\n");
			string tmp_fv = string(params.outputDir) + bovw_folder + to_string(params.num_cluster) + bovw_descriptor + ext;
			string t_path = tmp_fv;
			int idx = tmp_fv.find_last_of('.');
			t_path.insert(idx, "_info");
			descriptor_size = 0;
			dim = bovw_dim;
			if (LoadDescrBinaryInfo(t_path, &descriptor_size) < 0)
			{
				cout << "ERROR LOADING FV DESCRIPTORS";
				return;
			}
			db_size = imgs_url.size();
			float* fv_descriptors = (float*)malloc(descriptor_size*sizeof(float));
			LoadDescrBinary(tmp_fv, fv_descriptors, descriptor_size);

			int* index = (int*)malloc(sizeof(int)*db_size);
			for (int i = 0; i < db_size; i++)
				index[i] = i;
			float* model[nr_classes];
			float bias[nr_classes];
			int model_dim = 0;

			for (int i = 0; i < nr_classes; i++)
			{
				const char* class_name = i == 0 ? "fake" : "live";
				printf("Training svm for %s class\n", class_name);
				shuffle(index, db_size);
				double* labels = GetLabels(imgs_url, class_name);
				double* random_labels = (double*)malloc(sizeof(double)*db_size);
				float* random_fv = (float*)malloc(descriptor_size*sizeof(float));
				for (int j = 0; j < db_size; j++)
				{
					memcpy(&(random_fv[j*dim]), &(fv_descriptors[index[j] * dim]), sizeof(float)*dim);
					random_labels[j] = labels[index[j]];
				}

				//Setting svm paramters
				VlSvmDataset* dataset = vl_svmdataset_new(VL_TYPE_FLOAT, random_fv, dim, db_size);
				double lambda = (double)1 / (svm_C*db_size);
				VlSvm* svm = vl_svm_new_with_dataset(VlSvmSolverSdca, dataset, random_labels, lambda);
				vl_svm_set_bias_multiplier(svm, 1);
				vl_svm_set_loss(svm, VlSvmLossHinge);
				vl_svm_set_epsilon(svm, epsilon);
				vl_svm_set_max_num_iterations(svm, 50/lambda);

				vl_svm_train(svm);

				const double* m = vl_svm_get_model(svm);
				bias[i] = vl_svm_get_bias(svm);
				model_dim = vl_svm_get_dimension(svm);
				const VlSvmStatistics* statistics = vl_svm_get_statistics(svm);
				printf("loss = %f and dual_loss = %f\n", statistics->loss, statistics->dualLoss);
				switch (statistics->status)
				{
				case VlSvmStatusConverged:
					printf("Solver converged...\n");
					break;
				case VlSvmStatusMaxNumIterationsReached:
					printf("Iteration number reached...\n");
					break;
				}

				model[i] = (float*)malloc(sizeof(float)*model_dim);

				for (int j = 0; j < model_dim; j++)
					model[i][j] = m[j];

				vl_svmdataset_delete(dataset);
				vl_svm_delete(svm);
				free(random_fv);
				free(random_labels);
				free(labels);
			}
			free(fv_descriptors);
			free(index);
			save_model(model_path, model, bias, model_dim, nr_classes);
			for (int i = 0; i < nr_classes; i++)
				free(model[i]);
		}		
	}		
}
