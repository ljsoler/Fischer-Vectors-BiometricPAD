#include "bernoulliMixtureModel.h"
#include <omp.h>
#define INFINITY_F  MAXFLOAT
#define INFINITY_D  1.7976931348623158e+308

namespace clustering_and_indexing{

	namespace bmm{

BMM train_bmm_model(uchar* data, int numData, int numClusters, int featdim, int maxNumIterations)
{
	return fitBMM_log(data, numData, numClusters, featdim, maxNumIterations);
}

BMM fitBMM_log(uchar* data, int numData, int numClusters, int featdim, int maxNumIterations, int startMethod)
{
	int iteration, restarted;
	float previousLL = -INFINITY_F;
	float LL = -INFINITY_F;
	double time = 0;
	float margin = 0.25;
	float beta = 0.01;

	//DECLARE_TIMING(timer);

	// Initialize 
	float *priors = (float*)malloc(sizeof(float) * numClusters);
	float *means = (float*)malloc(sizeof(float) * numClusters * featdim);
	float* posteriors = (float*)malloc(sizeof(float) * numData * numClusters);

	if (startMethod == 1)	////////// Random //////////
	{
		for (int i = 0; i < numClusters; ++i) {
			priors[i] = (1.0 / numClusters);
		}

		RNG rng;
		vector<float> randN(numClusters*featdim);
		rng.fill(randN, RNG::UNIFORM, Scalar::all(0), Scalar::all(1));

		for (int j = 0; j < randN.size(); j++) {
			means[j] = randN[j] * (1 - 2 * margin) + margin; // Component means store concatenated each cluster of dimension featdim
		}

		//SaveDescrAsText("D:\\means.txt", means);
	}
	else if (startMethod == 2) ///// K-means + Random Prototype //////
	{
		//float* data_ = new float[numData * featdim];
		//for (int i = 0; i < numData*featdim ; i++){
		//	data_[i] = data[i];
		//}

		//free(data);
		//cout <<"free" <<endl;

		//// do-kmeans 
		//VlKMeans* kmeans = vl_kmeans_new (VL_TYPE_FLOAT, VlDistanceL1) ;
		//cout <<"new" <<endl;
		//vl_kmeans_set_algorithm (kmeans, VlKMeansLloyd) ;
		//cout <<"set alg" <<endl;
		//vl_kmeans_set_initialization ( kmeans, VlKMeansRandomSelection); 
		//cout <<"k-means init" <<endl;
		//vl_kmeans_set_max_num_iterations (kmeans, maxNumIterations) ;
		//double res =  vl_kmeans_cluster( kmeans, data_, featdim, numData, numClusters);
		//cout <<"k-means doed" <<endl;

		//float* centers;
		//centers = (float*)vl_kmeans_get_centers(kmeans) ;

		//vl_uint32 * assignments = (vl_uint32 *)vl_malloc(sizeof(vl_uint32) * numData);
		//vl_kmeans_quantize (kmeans, assignments, NULL, data_, numData) ;

		//cout <<"k-means quantize doed" <<endl;

		//int * numElemCluster = (int*)calloc(sizeof(int), numClusters);
		//for (int i = 0 ; i < numData ; i ++)
		//{
		//	numElemCluster[assignments[i]]++;		
		//}

		//RNG rng;
		//vector<float> randN(numClusters*featdim);
		//rng.fill(randN, RNG::UNIFORM, Scalar::all(0.3), Scalar::all(0.7));

		//float numData_ = numData;
		//for (int i = 0 ; i < numClusters ; ++i) 
		//{ 
		//	priors[i] = (numElemCluster[i] / numData_) ;

		//	for (int j = 0; j < featdim ; j++)
		//	{
		//		means[i*featdim+j] = 0.5 * randN[i*featdim+j] + 0.5 * centers[i*featdim+j];
		//	}
		//}

		//free(data_);
		//free(numElemCluster);
		//free(centers);
		//vl_free(assignments);

		cout << "training Kmeans" << endl;
		BMM_init_with_kmeans(data, means, priors, posteriors, numData, numClusters, featdim);

	}
	else if (startMethod == 3) ///// K-means matlab //////
	{
		FILE * fmeansb, *fpriorsb;
		string means_path = "E://TEORICO//EXPERIMENTS//DenseORB_matlab//C_mean_kmeans.txt";
		string priors_path = "E://TEORICO//EXPERIMENTS//DenseORB_matlab//priors_kmeans.txt";

		fmeansb = fopen(means_path.c_str(), "rb");
		/*if (fmeansb == NULL)
		return -2;*/

		fpriorsb = fopen(priors_path.c_str(), "rb");
		if (fpriorsb == NULL)
			cout << "error opening priors file" << endl;

		means = (float *)malloc(sizeof(float) * numClusters * featdim);
		priors = (float *)malloc(sizeof(float) * numClusters);

		fread(means, sizeof(float), numClusters * featdim, fmeansb);
		fread(priors, sizeof(float), numClusters, fpriorsb);

		fclose(fmeansb);
		fclose(fpriorsb);

	}
	else ///////// Random Prototype //////
	{

		RNG rng;
		vector<float> randN(numClusters*featdim);
		rng.fill(randN, RNG::UNIFORM, Scalar::all(0.3), Scalar::all(0.7));

		RNG rng_data;
		vector<int> randN_d(numClusters);
		rng.fill(randN_d, RNG::UNIFORM, Scalar::all(0), Scalar::all(numData - 1));

		for (int i = 0; i < numClusters; ++i)
		{
			priors[i] = (1.0 / numClusters);

			for (int j = 0; j < featdim; j++)
			{
				means[i*featdim + j] = 0.5 * randN[i*featdim + j] + 0.5 * data[randN_d[i] * featdim + j];
			}
		}

		/* Saving parameters of model */    //////////////////  QUITAR  /////////////////////
		string folderpath = "G:\\CENATAV\\TEORICO\\Databases\\YouTube Face Database\\data\\exploracion_biscay\\orb\\BMM\\iterations\\";

		/* Writing bernoulli means in a binary file */
		string means_path = folderpath + "/" + "means_" + to_string_(0) + "b.txt";
		FILE* fmeansb = fopen(means_path.c_str(), "wb");
		if (fmeansb == NULL)
			cout << "Error saving means" << endl;
		fwrite(means, sizeof(float), featdim * numClusters, fmeansb);
		fclose(fmeansb);

		/* Writing the priors of Spatial BMM in a binary file */
		string priors_path = folderpath + "/" + "priors_" + to_string_(0) + "b.txt";
		FILE* fpriorsb = fopen(priors_path.c_str(), "wb");
		if (fpriorsb == NULL)
			cout << "Error saving priors" << endl;
		fwrite(priors, sizeof(float), numClusters, fpriorsb);
		fclose(fpriorsb);

		//////////////////////////////////////////  FIN /////////////////////////////////////////////////////

	}
			
	for (iteration = 0; 1; ++iteration)
	{
		double eps;

		/* Expectation: assign data to Bernoulli modes and compute log-likelihood.
		*/

//		START_TIMING(timer);
	
		LL = get_data_posteriors(posteriors, numClusters, numData, priors, means, featdim, data);
				
		/*
		Check the termination conditions.
		*/

		eps = std::abs((LL - previousLL) / (LL));
		printf("BMM: em: iteration %d: loglikelihood = %f (variation = %f) (eps = %f)\n", iteration, LL, LL - previousLL, eps);

		if (iteration >= maxNumIterations)
		{
			printf("BMM: em: terminating because the maximum number of iterations (%d) has been reached.\n", maxNumIterations);
			break;
		}

		if ((iteration > 0) && (eps < 0.00001))
		{
			printf("BMM: em: terminating because the algorithm fully converged (log-likelihood variation = %f).\n", eps);
			break;
		}

		previousLL = LL;

		/* 
		Restart empty modes. 
		*/
		if (iteration > 1)
		{
			restarted = restart_empty_modes(posteriors, priors, means, data, numData, numClusters, featdim);
		}
			

		/* 
		Maximization: reestimate the BMM parameters. 
		*/
		bmm_maximization(posteriors, priors, means, data, numData, numClusters, featdim);

//		STOP_TIMING(timer);
//		SHOW_TIMING_(timer, "finished");


		///* Saving parameters of model */    //////////////////  QUITAR  /////////////////////
		//string folderpath = "G:\\CENATAV\\TEORICO\\Databases\\YouTube Face Database\\data\\exploracion_biscay\\orb\\BMM_featsel_infFS_128_0.2\\iterations\\";

		///* Writing posteriors in a binary file */
		//string posteriors_path = folderpath + "/" + "posteriors_" + to_string_(iteration + 1) + "b.txt";
		//FILE* fpstb = fopen(posteriors_path.c_str(), "wb");
		//if (fpstb == NULL)
		//	cout << "Error saving posteriors" << endl;
		//fwrite(posteriors, sizeof(float), numData * numClusters, fpstb);
		//fclose(fpstb);

		///* Writing bernoulli means in a binary file */
		//string means_path = folderpath + "/" + "means_" + to_string_(iteration + 1) + "b.txt";
		//FILE* fmeansb = fopen(means_path.c_str(), "wb");
		//if (fmeansb == NULL)
		//	cout << "Error saving means" << endl;
		//fwrite(means, sizeof(float), featdim * numClusters, fmeansb);
		//fclose(fmeansb);

		///* Writing the priors of Spatial BMM in a binary file */
		//string priors_path = folderpath + "/" + "priors_" + to_string_(iteration + 1) + "b.txt";
		//FILE* fpriorsb = fopen(priors_path.c_str(), "wb");
		//if (fpriorsb == NULL)
		//	cout << "Error saving priors" << endl;
		//fwrite(priors, sizeof(float), numClusters, fpriorsb);
		//fclose(fpriorsb);

		//////////////////////////////////////////  FIN /////////////////////////////////////////////////////
		
	}

	/* Saving parameters of model */    //////////////////  QUITAR  /////////////////////
	//string folderpath = "G:\\CENATAV\\TEORICO\\Databases\\YouTube Face Database\\data\\exploracion_biscay\\orb\\SpatialBMM_featsel_infFS_128_0.2_sigma_bound_center\\iterations\\";

	///* Writing posteriors in a binary file */
	//string posteriors_path = folderpath + "/" + "posteriors_" + to_string_(iteration + 1) + "b.txt";
	//FILE* fpstb = fopen(posteriors_path.c_str(), "wb");
	//if (fpstb == NULL)
	//	cout << "Error saving posteriors" << endl;
	//fwrite(posteriors, sizeof(float), numData * numClusters, fpstb);
	//fclose(fpstb);

	///* Writing bernoulli means in a binary file */
	//string means_path = folderpath + "/" + "means_" + to_string_(iteration + 1) + "b.txt";
	//FILE* fmeansb = fopen(means_path.c_str(), "wb");
	//if (fmeansb == NULL)
	//	cout << "Error saving means" << endl;
	//fwrite(means, sizeof(float), featdim * numClusters, fmeansb);
	//fclose(fmeansb);

	///* Writing the priors of Spatial BMM in a binary file */
	//string priors_path = folderpath + "/" + "priors_" + to_string_(iteration + 1) + "b.txt";
	//FILE* fpriorsb = fopen(priors_path.c_str(), "wb");
	//if (fpriorsb == NULL)
	//	cout << "Error saving priors" << endl;
	//fwrite(priors, sizeof(float), numClusters, fpriorsb);
	//fclose(fpriorsb);

	free(posteriors);

	BMM bmm_model;
	bmm_model.priors = priors;
	bmm_model.means = means;
	bmm_model.num_clusters = numClusters;
	bmm_model.feat_dimension = featdim;


	return bmm_model;

}

float logBer(int N_0, int N_1, int *X_0, int *X_1, float *A, float *B)
{
	float dot1 = 0, dot2 = 0, dot3 = 0, dot4 = 0;

	int i = 0;
	for(; i < N_0 - 3; i += 4) {
		dot1 += B[X_0[i]];
		dot2 += B[X_0[i+1]];
		dot3 += B[X_0[i+2]];
		dot4 += B[X_0[i+3]];
	}

	for (; i < N_0; i++) {
		dot1 += B[X_0[i]];
	}

	i = 0;
	for(; i < N_1 - 3; i += 4) {
		dot1 += A[X_1[i]];
		dot2 += A[X_1[i+1]];
		dot3 += A[X_1[i+2]];
		dot4 += A[X_1[i+3]];
	}

	for (; i < N_1; i++) {
		dot1 += A[X_1[i]];
	}

	return dot1 + dot2 + dot3 + dot4;
}

float get_data_posteriors(float* posteriors, int numClusters, int numData, const float * priors, const float * means, int featdim, const uchar * data)
{
	int i_d, i_cl, dim;
	float LL = 0;

	float *logWeights = (float*)malloc(sizeof(float)*numClusters);
	float *logMeans = (float*)malloc(sizeof(float)*numClusters * featdim);
	float *log1minusMean = (float*)malloc(sizeof(float)*numClusters * featdim);

	//#if defined(_OPENMP)
#pragma omp parallel for private(i_cl, dim) num_threads(vl_get_max_threads())
	//#endif
	for (i_cl = 0; i_cl < numClusters; ++i_cl)
	{
		float logSigma = 0;
		if (priors[i_cl] < BMM_MIN_PRIOR)
		{
			logWeights[i_cl] = -INFINITY_F;
		}
		else
		{
			logWeights[i_cl] = std::log(priors[i_cl]);
		}

		for (dim = 0; dim < featdim; ++dim)
		{
			logMeans[i_cl * featdim + dim] = log(means[i_cl * featdim + dim]);
			log1minusMean[i_cl * featdim + dim] = log(1. - means[i_cl * featdim + dim]);
		}		
	} /* end of parallel region */

	  //#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) reduction(+:LL) num_threads(vl_get_max_threads())
	  //#endif
	for (i_d = 0; i_d < numData; ++i_d)
	{
		float clusterPosteriorsSum = 0;
		float maxPosterior = -INFINITY_F;

		int* data0 = (int*)malloc(sizeof(int)*featdim);
		int* data1 = (int*)malloc(sizeof(int)*featdim);
		int c0 = 0, c1 = 0;
		for (int p = 0; p < featdim; p++)
		{
			*(data + i_d * featdim + p) == 0 ? data0[c0++] = p : data1[c1++] = p;
		}

		for (i_cl = 0; i_cl < numClusters; ++i_cl)
		{
			float pB = logBer(c0, c1, data0, data1, logMeans + i_cl * featdim, log1minusMean + i_cl * featdim);
			float p = logWeights[i_cl] + pB ;

			posteriors[i_cl + i_d * numClusters] = p;
			if (p > maxPosterior) { maxPosterior = p; }
		}

		free(data0);
		free(data1);

		for (i_cl = 0; i_cl < numClusters; ++i_cl)
		{
			float p = posteriors[i_cl + i_d * numClusters];
			p = exp(p - maxPosterior);
			posteriors[i_cl + i_d * numClusters] = p;
			clusterPosteriorsSum += p;
		}

		LL += log(clusterPosteriorsSum) + maxPosterior;

		for (i_cl = 0; i_cl < numClusters; ++i_cl)
		{
			posteriors[i_cl + i_d * numClusters] /= clusterPosteriorsSum;
		}
	}

	free(log1minusMean);
	free(logWeights);
	free(logMeans);

	return LL;
}

void BMM_init_with_kmeans(uchar* data, float* means, float* priors, float* posteriors, int numData, int numClusters, int featdim)
{
	int i_d;
	vl_uint32 * assignments = (vl_uint32 *)malloc(sizeof(vl_uint32) * numData);
		
	int dimension = featdim;
	float* data_ = (float*)malloc(sizeof(float)*numData * featdim);
	if(data_ == NULL)
		printf("No pudo crearlo \n");
	for (int i = 0; i < numData*featdim ; i++)	data_[i] = data[i];


	memset(means, 0, sizeof(float) * numClusters * featdim);
	memset(priors, 0, sizeof(float) * numClusters);
	memset(posteriors, 0, sizeof(float) * numClusters * numData);

	/* KMeans initalization object */

	vl_size ncomparisons = VL_MAX(numData / 4, 10);
	vl_size niter = 100;
	vl_size ntrees = 3;
	vl_size nrepetitions = 1;
	VlKMeansAlgorithm algorithm = VlKMeansElkan;
	VlKMeansInitialization initialization = VlKMeansPlusPlus;
	VlRand* random = vl_get_rand();
	vl_rand_seed(random, 1);
	VlKMeans * kmeansInitDefault = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
	vl_kmeans_set_initialization(kmeansInitDefault, initialization);
	vl_kmeans_set_max_num_iterations(kmeansInitDefault, niter);
	vl_kmeans_set_max_num_comparisons(kmeansInitDefault, ncomparisons);
	vl_kmeans_set_num_trees(kmeansInitDefault, ntrees);
	vl_kmeans_set_algorithm(kmeansInitDefault, algorithm);
	vl_kmeans_set_num_repetitions(kmeansInitDefault, nrepetitions);
	vl_kmeans_set_verbosity(kmeansInitDefault, 1);

	/* Use k-means to assign data to clusters */
	vl_kmeans_cluster(kmeansInitDefault, data_, dimension, numData, numClusters);
	vl_kmeans_quantize(kmeansInitDefault, assignments, NULL, data_, numData);

	/* Transform the k-means assignments in posteriors and estimates the mode parameters */
	for (i_d = 0; i_d < numData; i_d++) {
		((float*)posteriors)[assignments[i_d] + i_d * numClusters] = (float) 1.0;
	}
	free(data_);
	vl_kmeans_delete(kmeansInitDefault);
	/* Update cluster parameters */
	bmm_maximization(posteriors, priors, means, data, numData, numClusters, featdim);

	free(assignments);
	cout << " maximization finished" << endl;

//	/* Saving parameters of model */ //////////////////  QUITAR  /////////////////////
//	string folderpath = "G:\\CENATAV\\TEORICO\\Databases\\YouTube Face Database\\data\\exploracion_biscay\\orb\\SpatialBMM_featsel_infFS_128_0.2_sigma_bound_center\\iterations\\";
//
//	/* Writing posteriors in a binary file */
//	string posteriors_path = folderpath + "/" + "posteriors_" + to_string_(0) + "b.txt";
//	FILE* fpstb = fopen(posteriors_path.c_str(), "wb");
//	if (fpstb == NULL)
//		cout << "Error saving posteriors" << endl;
//	fwrite(posteriors, sizeof(float), numData * numClusters, fpstb);
//	fclose(fpstb);

	/* Writing bernoulli means in a binary file */
//	string means_path = folderpath + "/" + "means_" + to_string_(0) + "b.txt";
//	FILE* fmeansb = fopen(means_path.c_str(), "wb");
//	if (fmeansb == NULL)
//		cout << "Error saving means" << endl;
//	fwrite(means, sizeof(float), featdim * numClusters, fmeansb);
//	fclose(fmeansb);

	/* Writing the priors of Spatial BMM in a binary file */
//	string priors_path = folderpath + "/" + "priors_" + to_string_(0) + "b.txt";
//	FILE* fpriorsb = fopen(priors_path.c_str(), "wb");
//	if (fpriorsb == NULL)
//		cout << "Error saving priors" << endl;
//	fwrite(priors, sizeof(float), numClusters, fpriorsb);
//	fclose(fpriorsb);
//
//
//	cout << "kmeans initialization saved " << endl;

	/////////////////////////////////////  FIN //////////////////////////////////////
}

void bmm_maximization(float* posteriors, float* priors, float* means, const uchar *data, int numData, int numClusters, int featdim)
{
	int i_d, i_cl, dim;

	memset(priors, 0, sizeof(float) * numClusters);
	memset(means, 0, sizeof(float) * featdim * numClusters);
	
	//#if defined(_OPENMP)
#pragma omp parallel default(shared) private(i_d, i_cl, dim) num_threads(omp_get_max_threads())                     
	//#endif
	{
		float *clusterPosteriorSum_, *means_;

		//#if defined(_OPENMP)
#pragma omp critical
		//#endif
		{
			clusterPosteriorSum_ = (float*)calloc(sizeof(float), numClusters);
			means_ = (float*)calloc(sizeof(float), featdim * numClusters);
		}

		//#if defined(_OPENMP)
#pragma omp for
		//#endif
		for (i_d = 0; i_d < (signed)numData; ++i_d)
		{
			for (i_cl = 0; i_cl < (signed)numClusters; ++i_cl)
			{
				float p = posteriors[i_cl + i_d * numClusters];

				/* skip very small associations for speed */
				if (p < BMM_MIN_POSTERIOR / numClusters) { continue; }

				clusterPosteriorSum_[i_cl] += p;

				for (dim = 0; dim < featdim; ++dim)
				{
					uchar x = data[i_d * featdim + dim];
					means_[i_cl * featdim + dim] += p * x;				
				}				
			}
		}

		//#if defined(_OPENMP)
#pragma omp critical
		//#endif
		{
			for (i_cl = 0; i_cl < (signed)numClusters; ++i_cl)
			{
				priors[i_cl] += clusterPosteriorSum_[i_cl];
				for (dim = 0; dim < featdim; dim++)
				{
					means[i_cl * featdim + dim] += means_[i_cl * featdim + dim];					
				}				
			}

			free(means_);
			free(clusterPosteriorSum_);
		}
	} /* end parallel section */

	  /* at this stage priors[] contains the total mass of each cluster */
	for (i_cl = 0; i_cl < numClusters; ++i_cl)
	{
		//priors[i_cl] = clusterPosteriorSum_[i_cl] / numData; // aki el GMM divide entre la suma de clusterPosteriorSum_)
		float mass = priors[i_cl];

		// pensar aki como seria el umbral para el caso de nosotros el if(mass >= 1e-6 / numClusters) 
		if (mass >= 1e-6 / numClusters)
		{
			for (dim = 0; dim < featdim; ++dim)
			{
				means[i_cl * featdim + dim] /= mass;
				if (means[i_cl * featdim + dim] > max_phit)
					means[i_cl * featdim + dim] = max_phit;
				else if (means[i_cl * featdim + dim] < min_phit)
					means[i_cl * featdim + dim] = min_phit;				
			}			
		}
	}	


	{
		float sum = 0;
		for (i_cl = 0; i_cl < (signed)numClusters; ++i_cl) {
			sum += priors[i_cl];
		}
		sum = MAX(sum, 1e-12);
		for (i_cl = 0; i_cl < (signed)numClusters; ++i_cl) {
			priors[i_cl] /= sum;
		}
	}
}

int SaveBMMResultsAsText(string folderpath, string exp_name, BMM bmm_model)
{
	FILE * ofp, *fmeans, *fpriors;
	int dimension = bmm_model.feat_dimension;
	int K = bmm_model.num_clusters;

	float const * means = bmm_model.means;
	float const * priors = bmm_model.priors;

	string bmm_info = folderpath + "/" + exp_name + "_gmm_info.txt";
	string means_path = folderpath + "/" + exp_name + "_means_" + to_string_(K) + ".txt";
	string priors_path = folderpath + "/" + exp_name + "_priors_" + to_string_(K) + ".txt";

	ofp = fopen(bmm_info.c_str(), "w"); //  [means, sigmas, weights]
	if (ofp == NULL)
		return -1;

	fmeans = fopen(means_path.c_str(), "w");
	if (fmeans == NULL)
		return -2;

	fpriors = fopen(priors_path.c_str(), "w");
	if (fpriors == NULL)
		return -3;

	fprintf(ofp, "numclusters %d \n", K);
	fprintf(ofp, "dimension %d \n", dimension);

	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			fprintf(fmeans, "%f ", means[i * dimension + j]);
		}

		fprintf(fpriors, "%f ", priors[i]);
		fprintf(fmeans, "\n");
		fprintf(fpriors, "\n");
	}

	fclose(ofp);
	fclose(fmeans);
	fclose(fpriors);

	return 1;

}

int SaveBMMResultsBinary(string folderpath, string exp_name, BMM bmm_model)
{

	FILE * ofp, *fmeansb, *fpriorsb;
	int dimension = bmm_model.feat_dimension;
	int K = bmm_model.num_clusters;

	float const * means = bmm_model.means;
	float const * priors = bmm_model.priors;

	string bmm_info = folderpath + "/" + exp_name + "_gmm_info.txt";
	string means_path = folderpath + "/" + exp_name + "_means_" + to_string_(K) + "b.txt";
	string priors_path = folderpath + "/" + exp_name + "_priors_" + to_string_(K) + "b.txt";

	ofp = fopen(bmm_info.c_str(), "w"); //  [means, sigmas, weights]
	if (ofp == NULL)
		return -1;

	fmeansb = fopen(means_path.c_str(), "wb");
	if (fmeansb == NULL)
		return -2;

	fpriorsb = fopen(priors_path.c_str(), "wb");
	if (fpriorsb == NULL)
		return -4;

	fwrite(means, sizeof(float), dimension * K, fmeansb);
	fwrite(priors, sizeof(float), K, fpriorsb);

	fclose(fmeansb);
	fclose(fpriorsb);

	fprintf(ofp, "numclusters %d \n", K);
	fprintf(ofp, "dimension %d \n", dimension);

	fclose(ofp);

	return 1;

}

int LoadBMMResultsBinary(string folderpath, string exp_name, float*& means, float*& priors, int &K, int &dimension)
{
	string gmm_info = folderpath + "/" + exp_name + "_gmm_info" + ".txt";

	FILE * ofp, *fmeansb, *fpriorsb;
	ofp = fopen(gmm_info.c_str(), "r"); //  [means, weights]
	if (ofp == NULL)
		return -1;

	int datatype;
	char buffer[100];

	fscanf(ofp, "%s", buffer);  fscanf(ofp, "%d", &K);
	fscanf(ofp, "%s", buffer);  fscanf(ofp, "%d", &dimension);

	string means_path = folderpath + "/" + exp_name + "_means_" + to_string_((int)K) + "b.txt";
	string priors_path = folderpath + "/" + exp_name + "_priors_" + to_string_((int)K) + "b.txt";

	fmeansb = fopen(means_path.c_str(), "rb");
	if (fmeansb == NULL)
		return -2;

	fpriorsb = fopen(priors_path.c_str(), "rb");
	if (fpriorsb == NULL)
		return -3;

	means = (float *)malloc(sizeof(float) * K * dimension);
	priors = (float *)malloc(sizeof(float) * K);

	fread(means, sizeof(float), dimension * K, fmeansb);
	fread(priors, sizeof(float), K, fpriorsb);

	fclose(fmeansb);
	fclose(fpriorsb);
	fclose(ofp);

	return 1;
}

int SaveFisherAsText(string path, double* descr, int dimension)
{
	std::ofstream fout(path);
	if (fout.fail()) {
		cout << "ERROR opening file " << path << endl;
		return -1;
	}

	for (int f = 0; f < dimension; f++)
	{
		fout << descr[f] << endl;
	}

	return 1;
}

int SaveFisherAsText(string path, float* descr, int dimension)
{
	std::ofstream fout(path);
	if (fout.fail()) {
		cout << "ERROR opening file " << path << endl;
		return -1;
	}

	for (int f = 0; f < dimension; f++)
	{
		fout << descr[f] << endl;
	}

	return 1;
}

int fisher_encode(float* enc, float* means, float* priors, uchar* data, int numData, int featdim, int numClusters, int flag_norm, int flag_square, int flag_fast)
{
	int dim, sdim, i_cl, i_d, numTerms = 0;
	float* posteriors, *sqrtInvSigma, *sum_pi_means, *sum_pi_ones_minus_mean;

	int fv_dim = numClusters * featdim;

	assert(numClusters >= 1);
	assert(featdim >= 1);

	posteriors = (float*)malloc(sizeof(float) * numClusters * numData);
	sum_pi_means = (float*)calloc(sizeof(float), featdim);
	sum_pi_ones_minus_mean = (float*)calloc(sizeof(float), featdim);
	memset(enc, 0, sizeof(float) * fv_dim);

	for (i_cl = 0; i_cl < (signed)numClusters; ++i_cl)
	{
		for (dim = 0; dim < featdim; dim++)
		{ 
			sum_pi_means[dim] += priors[i_cl] * means[i_cl * featdim + dim];
			sum_pi_ones_minus_mean[dim] += priors[i_cl] * (1 - means[i_cl*featdim + dim]);
		}
	}

	float LL = get_data_posteriors(posteriors, numClusters, numData, priors, means, featdim, data);

	/* sparsify posterior assignments with the FAST option */
	if (flag_fast)
	{
		for (i_d = 0; i_d < numData; i_d++)
		{
			/* find largest posterior assignment for datum i_d */
			int best = 0;
			float bestValue = posteriors[i_d * numClusters];
			for (i_cl = 1; i_cl < numClusters; ++i_cl)
			{
				float p = posteriors[i_cl + i_d * numClusters];
				if (p > bestValue)
				{
					bestValue = p;
					best = i_cl;
				}
			}
			/* make all posterior assignments zero but the best one */
			for (i_cl = 0; i_cl < numClusters; ++i_cl)
			{
				posteriors[i_cl + i_d * numClusters] = (float)(i_cl == best);
			}
		}
	}

	//#if defined(_OPENMP)
#pragma omp parallel for default(shared) private(i_cl, i_d, dim, sdim) num_threads(omp_get_max_threads()) reduction(+:numTerms)
	//#endif
	for (i_cl = 0; i_cl < numClusters; ++i_cl)
	{
		float uprefix;
		float vprefix;

		float * uk = enc + i_cl * featdim;

		if (priors[i_cl] < BMM_MIN_PRIOR) { continue; }

		for (i_d = 0; i_d < numData; i_d++)
		{
			float p = posteriors[i_cl + i_d * numClusters];
			if (p < 1e-6) continue;
			numTerms += 1;
			for (dim = 0; dim < featdim; dim++)
			{
				uchar d = data[i_d * featdim + dim];
				float m = means[i_cl * featdim + dim];
				float frac = pow(-1, (1. - d)) / (m*d + (1 - m) * (1 - d));
				*(uk + dim) += frac * p;
			}			
		}

		if (numData > 0)
		{
			uprefix = 1 / (numData * sqrt(priors[i_cl]));
			for (dim = 0; dim < featdim; dim++)
			{
				float a = sum_pi_means[dim] / (means[i_cl * featdim + dim] * means[i_cl * featdim + dim]);
				float b = sum_pi_ones_minus_mean[dim] / ((1 - means[i_cl * featdim + dim]) * (1 - means[i_cl * featdim + dim]));
				double div = a + b;
				double uprefix1 = uprefix / sqrt(div);
				*(uk + dim) = *(uk + dim) * uprefix1;
			}			
		}
	}

	free(posteriors);
	free(sum_pi_means);
	free(sum_pi_ones_minus_mean);

	if (flag_square)
	{
		for (dim = 0; dim < fv_dim; dim++)
		{
			float z = enc[dim];
			if (z >= 0)
			{
				enc[dim] = sqrt(z);
			}
			else
			{
				enc[dim] = -sqrt(-z);
			}
		}
	}

	if (flag_norm)
	{
		float n = 0;
		for (dim = 0; dim < fv_dim; dim++)
		{
			float z = enc[dim];
			n += z * z;
		}
		n = sqrt(n);
		n = MAX(n, 1e-12);
		for (dim = 0; dim < fv_dim; dim++)
		{
			enc[dim] /= n;
		}
	}

	return numTerms;
}

int restart_empty_modes(float* posteriors, float* priors, float* means, uchar *data, int numData, int numClusters, int featdim)
{
	int i_cl, j_cl, i_d, d;

	float* mass = (float*)calloc(sizeof(float), numClusters);

	if (numClusters <= 1) { return 0; }

	/* compute statistics */
	int i, k;
	int numNullAssignments = 0;
	for (i = 0; i < numData; ++i)
	{
		for (k = 0; k < numClusters; ++k)
		{
			float p = posteriors[k + i * numClusters];
			mass[k] += p;
			if (p < BMM_MIN_POSTERIOR)
			{
				numNullAssignments++;
			}
		}
	}
	printf("bmm: sparsity of data posterior: %.1f%%\n", (float)numNullAssignments / (numData * numClusters) * 100);

	/* search for cluster with negligible weight and reassign them to fat clusters */
	for (i_cl = 0; i_cl < numClusters; ++i_cl)
	{
		double size = -INFINITY_D;
		int best = -1;

		if (mass[i_cl] >= BMM_MIN_POSTERIOR * MAX(1.0, (float)numData / numClusters))
		{
			continue;
		}

		printf("bmm: mode %d is nearly empty (mass %f)\n", i_cl, mass[i_cl]);

		/*
		Search for the BMM components that (approximately)
		maximally contribute to make the negative log-likelihood of the data
		large. Then split the worst offender.

		To do so, we approximate the exptected log-likelihood of the SBMM:

		E[-log(f(x))] = H(f) = - log \int f(x) log f(x)

		where the density f(x) = sum_k wk gk(x) is a BMM. This is intractable
		but it is easy to approximate if we suppose that supp gk is disjoint with
		supp gq for all components k ~= q. In this canse

		H(f) ~= sum_k [ - wk log(wk) + wk H(gk) ]

		where H(gk) is the entropy of component k taken alone. The entropy of
		the latter is given by:

		H(gk) =  - ( sum_{i=0}^D p_i log(p_i) + (1-p_i) log(1-p_i) )

		*/

		for (j_cl = 0; j_cl < numClusters; ++j_cl)
		{
			double size_ = 0;
			if (priors[j_cl] < BMM_MIN_PRIOR)
			{
				continue;
			}
			for (d = 0; d < featdim; ++d)
			{
				float p = means[j_cl * featdim + d];
				size_ += (-1) * p * log(p) - (1 - p) * log(1 - p);
			}
			size_ = priors[j_cl] * (size_ - log(priors[j_cl]));

			printf("bmm: mode %d: prior %f, mass %f, entropy contribution %f\n",
				j_cl, priors[j_cl], mass[j_cl], size_);

			if (size_ > size)
			{
				size = size_;
				best = j_cl;
			}
		}

		j_cl = best;

		if (j_cl == i_cl || j_cl < 0)
		{
			printf("bmm: mode %d is empty, "
				"but no other mode to split could be found\n", i_cl);
			continue;
		}

		printf("bmm: reinitializing empty mode %d with mode %d (prior %f, mass %f, score %f)\n",
			i_cl, j_cl, priors[j_cl], mass[j_cl], size);

		/*
		Search for the dimension with maximum variance.
		*/

		size = -INFINITY_D;
		best = -1;

		for (d = 0; d < featdim; ++d)
		{
			float sigma2 = means[j_cl * featdim + d] * (1 - means[j_cl * featdim + d]);
			if (sigma2 > size)
			{
				size = sigma2;
				best = d;
			}
		}

		/*
		Reassign points j_cl (mode to split) to i_cl (empty mode).
		*/
		{
			float mu = means[best + j_cl * featdim];

			for (i_d = 0; i_d < numData; ++i_d)
			{
				float p = posteriors[j_cl + numClusters * i_d];
				float q = posteriors[i_cl + numClusters * i_d]; /* ~= 0 */

				if (data[best + i_d * featdim] < mu)
				{
					/* assign this point to i_cl */
					posteriors[i_cl + numClusters * i_d] = p + q;
					posteriors[j_cl + numClusters * i_d] = 0;
				}
				else
				{
					/* assign this point to j_cl */
					posteriors[i_cl + numClusters * i_d] = 0;
					posteriors[j_cl + numClusters * i_d] = p + q;
				}
			}
		}

		/*
		Re-estimate.
		*/
		bmm_maximization(posteriors, priors, means, data, numData, numClusters, featdim);

	}
}

	}
};

