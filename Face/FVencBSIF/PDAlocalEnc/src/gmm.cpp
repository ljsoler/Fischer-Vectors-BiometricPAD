#include "gmm.h"
#include "common.h"
#include <stdio.h>

using namespace std;

namespace clustering_and_indexing{

	namespace gmm{

		VlGMM* train_gmm_model(float* desc, int total_data, int dimension, int numclusters, int max_iter = 100, int max_iterkmean = 100)
		{
			// create a GMM object and cluster input data to get means, covariances
			// and priors of the estimated mixture	
			/*double energy;
			double * centers;*/
			float* data = desc;

			//VlKMeans* kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
			//vl_kmeans_set_verbosity(kmeans, 1);
			//vl_kmeans_set_algorithm(kmeans, VlKMeansElkan);
			//vl_kmeans_init_centers_with_rand_data (kmeans, data, dimension, total_data, numclusters) ;
			//vl_kmeans_set_initialization(kmeans, VlKMeansRandomSelection);
			// Run at most 100 iterations of cluster refinement using Lloyd algorithm
			//vl_kmeans_set_max_num_iterations(kmeans, max_iterkmean);
			//vl_kmeans_refine_centers (kmeans, data, total_data) ;
			// Obtain the energy of the solution
			//energy = vl_kmeans_get_energy(kmeans) ;
			//// Obtain the cluster centers
			//centers = vl_kmeans_get_centers(kmeans) ;
			//double res = vl_kmeans_cluster(kmeans, data, dimension, total_data, numclusters);

			float*means = (float*)malloc(sizeof(float)*dimension);	
			for (int i = 0; i < dimension; i++)
			{
				means[i] = 0;
				for (int j = 0; j < total_data; j++)
					means[i] += data[j*dimension + i];
				means[i] /= (float)total_data;
			}		

			double variance = -100000000.0;
			for(int i = 0; i < dimension; i++)
			{
				double temp = 0;
				for(int j = 0; j < total_data; j++)
					temp += ((data[j*dimension + i] - means[i])*(data[j*dimension + i] - means[i]));

				temp /= (double)(total_data - 1);
				if(temp > variance)
					variance = temp;
			}
			free(means);
			VlRand* random = vl_get_rand();
			vl_rand_seed(random, 1);
			VlGMM* gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numclusters);
			vl_gmm_set_verbosity(gmm, 1);
			vl_gmm_set_max_num_iterations(gmm, max_iter);			
			vl_gmm_set_initialization(gmm, VlGMMKMeans);
			vl_gmm_set_covariance_lower_bound(gmm, variance*0.0001);
			//vl_gmm_init_with_kmeans(gmm, data, total_data, kmeans);
			double d1 = vl_gmm_cluster(gmm, data, total_data);
			//vl_kmeans_delete(kmeans);
			return gmm;
			//vl_free(data);		
		}

		void save_gmm_results(char * _path, VlGMM * gmm)
		{
			vl_size d, cIdx;
			vl_uindex i_d;

			FILE * ofp, *fmeans, *fsigmas, *fweights, *fposteriors;
			FILE * ofpb, *fmeansb, *fsigmasb, *fweightsb, *fgeneraldata;
			vl_size dimension = vl_gmm_get_dimension(gmm);
			vl_size numClusters = vl_gmm_get_num_clusters(gmm);
			vl_type dataType = vl_gmm_get_data_type(gmm);
			vl_size numData = vl_gmm_get_num_data(gmm);

			float const * sigmas = (float const *)vl_gmm_get_covariances(gmm);
			float const * means = (float const *)vl_gmm_get_means(gmm);
			float const * weights = (float const *)vl_gmm_get_priors(gmm);
			//double const * posteriors = (double const *)vl_gmm_get_posteriors(gmm) ;

			char * path = NULL;
			char * gmm_file = "gmm.txt";  //todo junto
			//char * means_file = "JanusFisher/means.txt";
			//char * sigmas_file = "JanusFisher/sigmas.txt";
			//char * weights_file = "JanusFisher/weights.txt";
			//char * posteriors_file = "JanusFisher/posteriors.txt";

			/*path = new char[strlen(_path) + strlen(gmm_file) + 1];
			strcpy(path, _path);
			strcat(path, gmm_file);*/

			ofp = fopen(_path, "w"); // fichero q lo tiene todo concatenado [means, sigmas, weights]
			fmeans = fopen("means.txt", "w");
			fsigmas = fopen("sigmas.txt", "w");
			fweights = fopen("weights.txt", "w");
			//fposteriors = fopen("./resources/JanusFisher/posteriors.txt", "w");

			fmeansb = fopen("meansb.txt", "wb");
			fsigmasb = fopen("sigmasb.txt", "wb");
			fweightsb = fopen("weightsb.txt", "wb");

			fwrite(means, sizeof(float), dimension * numClusters, fmeansb);
			fwrite(sigmas, sizeof(float), dimension * numClusters, fsigmasb);
			fwrite(weights, sizeof(float), numClusters, fweightsb);

			fclose(fmeansb);
			fclose(fsigmasb);
			fclose(fweightsb);

			fprintf(ofp, "numclusters %d \n", (int)numClusters);
			fprintf(ofp, "dimension %d \n", (int)dimension);
			fprintf(ofp, "datatype %d \n", (int)dataType);

			for (cIdx = 0; cIdx < numClusters; cIdx++)
			{
				if (dataType == VL_TYPE_DOUBLE)
				{
					for (d = 0; d < dimension; d++)
					{
						double val = ((double*)means)[cIdx*dimension + d];
						fprintf(ofp, "%f ", val);
						fprintf(fmeans, "%f ", ((double*)means)[cIdx*dimension + d]);
					}

					for (d = 0; d < dimension; d++)
					{
						double val = ((double*)sigmas)[cIdx*dimension + d];
						fprintf(ofp, "%f ", val);
						fprintf(fsigmas, "%f ", ((double*)sigmas)[cIdx*dimension + d]);
					}

					fprintf(ofp, "%f ", ((double*)weights)[cIdx]);
					fprintf(fweights, "%f ", ((double*)weights)[cIdx]);

					fprintf(ofp, "\n");
					fprintf(fmeans, "\n");
					fprintf(fsigmas, "\n");
					fprintf(fweights, "\n");
					//fprintf(fposteriors, "\n");
				}
				else
				{
					for (d = 0; d < dimension; d++)
					{
						fprintf(ofp, "%f ", ((float*)means)[cIdx*dimension + d]);
						fprintf(fmeans, "%f ", ((float*)means)[cIdx*dimension + d]);
					}

					for (d = 0; d < dimension; d++)
					{
						fprintf(ofp, "%f ", ((float*)sigmas)[cIdx*dimension + d]);
						fprintf(fsigmas, "%f ", ((float*)sigmas)[cIdx*dimension + d]);
					}

					fprintf(ofp, "%f ", ((float*)weights)[cIdx]);
					fprintf(fweights, "%f ", ((float*)weights)[cIdx]);

					fprintf(ofp, "\n");
					fprintf(fmeans, "\n");
					fprintf(fsigmas, "\n");
					fprintf(fweights, "\n");
					//fprintf(fposteriors, "\n");
				}
			}
			fclose(ofp);
			fclose(fmeans);
			fclose(fsigmas);
			fclose(fweights);
			//fclose (fposteriors);
		}

		int save_gmm_results_binary(string folderpath, string exp_name, VlGMM * gmm)
		{

			FILE * ofp, *fmeansb, *fsigmasb, *fweightsb;
			vl_size dimension = vl_gmm_get_dimension(gmm);
			vl_size K = vl_gmm_get_num_clusters(gmm);
			vl_type dataType = vl_gmm_get_data_type(gmm);
			vl_size numData = vl_gmm_get_num_data(gmm);

			float const * sigmas = (float const *)vl_gmm_get_covariances(gmm);
			float const * means = (float const *)vl_gmm_get_means(gmm);
			float const * weights = (float const *)vl_gmm_get_priors(gmm);

			string gmm_info = folderpath + "/" + exp_name + "_gmm_info.txt";
			string means_path = folderpath + "/" + exp_name + "_means_" + to_string_((int)K) + "b.txt";
			string sigmas_path = folderpath + "/" + exp_name + "_sigmas_" + to_string_((int)K) + "b.txt";
			string weights_path = folderpath + "/" + exp_name + "_weights_" + to_string_((int)K) + "b.txt";

			ofp = fopen(gmm_info.c_str(), "w"); //  [means, sigmas, weights]
			if (ofp == NULL)
				return -1;

			fmeansb = fopen(means_path.c_str(), "wb");
			if (fmeansb == NULL)
				return -2;

			fsigmasb = fopen(sigmas_path.c_str(), "wb");
			if (fsigmasb == NULL)
				return -3;

			fweightsb = fopen(weights_path.c_str(), "wb");
			if (fweightsb == NULL)
				return -4;

			fwrite(means, sizeof(float), dimension * K, fmeansb);
			fwrite(sigmas, sizeof(float), dimension * K, fsigmasb);
			fwrite(weights, sizeof(float), K, fweightsb);

			fclose(fmeansb);
			fclose(fsigmasb);
			fclose(fweightsb);

			fprintf(ofp, "numclusters %d \n", (int)K);
			fprintf(ofp, "dimension %d \n", (int)dimension);
			fprintf(ofp, "datatype %d \n", (int)dataType);

			fclose(ofp);

			return 1;

		}

		int save_gmm_results_as_text(string folderpath, string exp_name, VlGMM * gmm)
		{
			FILE * ofp, *fmeans, *fsigmas, *fweights;
			vl_size dimension = vl_gmm_get_dimension(gmm);
			vl_size K = vl_gmm_get_num_clusters(gmm);
			vl_type dataType = vl_gmm_get_data_type(gmm);
			vl_size numData = vl_gmm_get_num_data(gmm);

			float const * sigmas = (float const *)vl_gmm_get_covariances(gmm);
			float const * means = (float const *)vl_gmm_get_means(gmm);
			float const * weights = (float const *)vl_gmm_get_priors(gmm);

			string gmm_info = folderpath + "/" + exp_name + "_gmm_info.txt";
			string means_path = folderpath + "/" + exp_name + "_means_" + to_string_((int)K) + ".txt";
			string sigmas_path = folderpath + "/" + exp_name + "_sigmas_" + to_string_((int)K) + ".txt";
			string weights_path = folderpath + "/" + exp_name + "_weights_" + to_string_((int)K) + ".txt";

			ofp = fopen(gmm_info.c_str(), "w"); //  [means, sigmas, weights]
			if (ofp == NULL)
				return -1;

			fmeans = fopen(means_path.c_str(), "w");
			if (fmeans == NULL)
				return -2;

			fsigmas = fopen(sigmas_path.c_str(), "w");
			if (fsigmas == NULL)
				return -3;

			fweights = fopen(weights_path.c_str(), "w");
			if (fweights == NULL)
				return -4;

			fprintf(ofp, "numclusters %d \n", (int)K);
			fprintf(ofp, "dimension %d \n", (int)dimension);
			fprintf(ofp, "datatype %d \n", (int)dataType);

			for (int i = 0; i < K; i++)
			{
				for (int j = 0; j < dimension; j++)
				{
					fprintf(fmeans, "%f ", ((float*)means)[i * dimension + j]);
					fprintf(fsigmas, "%f ", ((float*)sigmas)[i * dimension + j]);
				}

				fprintf(fweights, "%f ", ((float*)weights)[i]);
				fprintf(fmeans, "\n");
				fprintf(fsigmas, "\n");
				fprintf(fweights, "\n");
			}

			fclose(ofp);
			fclose(fmeans);
			fclose(fsigmas);
			fclose(fweights);

			return 1;

		}

		int load_gmm_results_binary(string folderpath, string exp_name, float*& means, float*& sigmas, float*& weights, int &K, int &dimension)
		{
			string gmm_info = folderpath + exp_name + "_gmm_info" + ".txt";

			FILE * ofp, *fmeansb, *fsigmasb, *fweightsb;
			cout << gmm_info << endl;
			ofp = fopen(gmm_info.c_str(), "r"); //  [means, sigmas, weights]
			if (ofp == NULL)
				return -1;

			int datatype;
			char buffer[100];

			fscanf(ofp, "%s", buffer);  fscanf(ofp, "%d", &K);
			fscanf(ofp, "%s", buffer);  fscanf(ofp, "%d", &dimension);
			fscanf(ofp, "%s", buffer);  fscanf(ofp, "%d", &datatype);

			string means_path = folderpath +  exp_name + "_means_" + to_string_((int)K) + "b.txt";
			string sigmas_path = folderpath + exp_name + "_sigmas_" + to_string_((int)K) + "b.txt";
			string weights_path = folderpath + exp_name + "_weights_" + to_string_((int)K) + "b.txt";

			fmeansb = fopen(means_path.c_str(), "rb");
			if (fmeansb == NULL)
				return -2;

			fsigmasb = fopen(sigmas_path.c_str(), "rb");
			if (fsigmasb == NULL)
				return -3;

			fweightsb = fopen(weights_path.c_str(), "rb");
			if (fweightsb == NULL)
				return -4;

			sigmas = (float *)malloc(sizeof(float)* K * dimension);
			means = (float *)malloc(sizeof(float)* K * dimension);
			weights = (float *)malloc(sizeof(float)* K);

			fread(means, sizeof(float), dimension * K, fmeansb);
			fread(sigmas, sizeof(float), dimension * K, fsigmasb);
			fread(weights, sizeof(float), K, fweightsb);

			fclose(fmeansb);
			fclose(fsigmasb);
			fclose(fweightsb);
			fclose(ofp);

			return 1;
		}

		int load_gmm_results_binary(string means_path, string sigmas_path, string weights_path, float*& means, float*& sigmas, float*& weights, int &K, int &dimension)
		{
			FILE *fmeansb, *fsigmasb, *fweightsb;

			fmeansb = fopen(means_path.c_str(), "rb");
			if (fmeansb == NULL)
				return -2;

			fsigmasb = fopen(sigmas_path.c_str(), "rb");
			if (fsigmasb == NULL)
				return -3;

			fweightsb = fopen(weights_path.c_str(), "rb");
			if (fweightsb == NULL)
				return -4;

			sigmas = (float *)malloc(sizeof(float)* K * dimension);
			means = (float *)malloc(sizeof(float)* K * dimension);
			weights = (float *)malloc(sizeof(float)* K);

			fread(means, sizeof(float), dimension * K, fmeansb);
			fread(sigmas, sizeof(float), dimension * K, fsigmasb);
			fread(weights, sizeof(float), K, fweightsb);

			fclose(fmeansb);
			fclose(fsigmasb);
			fclose(fweightsb);

			return 1;
		}

		void load_gmm_data(string folder_path, float *&means, float *&sigmas, float *&weights, int &K, int &dimension)
		{
			string gmm_filename = folder_path + "/GMM_fv_info.txt";
			FILE* ofp = fopen(gmm_filename.c_str(), "r");
			int datatype;
			char buffer[100];
			fscanf(ofp, "%s", buffer);  fscanf(ofp, "%d", &K);
			fscanf(ofp, "%s", buffer);  fscanf(ofp, "%d", &dimension);
			fscanf(ofp, "%s", buffer);  fscanf(ofp, "%d", &datatype);

			string means_filename = folder_path + "/GMM_fv_means_" + to_string((long long)K) + "b.txt";
			string sigmas_filename = folder_path + "/GMM_fv_sigmas_" + to_string((long long)K) + "b.txt";
			string weights_filename = folder_path + "/GMM_fv_weights_" + to_string((long long)K) + "b.txt";

			load_gmm_results_binary(means_filename, sigmas_filename, weights_filename, means, sigmas, weights, K, dimension);
		}
	}
}
