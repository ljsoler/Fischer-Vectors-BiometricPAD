#include <opencv2/opencv.hpp>
//#include "mytimer.h"
#include "common.h"

#define BMM_MIN_PRIOR		1e-6
#define BMM_MIN_POSTERIOR	1e-2
#define min_phit			1e-4
#define max_phit			1 - 1e-4

#include <vl/gmm.h>

struct BMM {
	float* priors;
	float* means;
	int num_clusters;
	int feat_dimension;
	double LL;
};

namespace clustering_and_indexing{

	namespace bmm{

	BMM train_bmm_model(uchar* data, int numData, int numClusters, int featdim, int maxNumIterations);

	BMM fitBMM_log(uchar* data, int numData, int numClusters, int featdim, int maxNumIterations, int startMethod = 2);

	void BMM_init_with_kmeans(uchar* data, float* means, float* priors, float* posteriors, int numData, int numClusters, int featdim);

	float get_data_posteriors(float* posteriors, int numClusters, int numData, const float * priors, const float * means, int featdim, const uchar * data);

	void bmm_maximization(float* posteriors, float* priors, float* means, const uchar *data, int numData, int numClusters, int featdim);

	int SaveBMMResultsAsText(string folderpath, string exp_name, BMM bmm_model);

	int SaveBMMResultsBinary(string folderpath, string exp_name, BMM bmm_model);

	int LoadBMMResultsBinary(string folderpath, string exp_name, float*& means, float*& priors, int &K, int &dimension);

	int SaveFisherAsText(string path, double* descr, int dimension);
	int SaveFisherAsText(string path, float* descr, int dimension);

	int fisher_encode(float* enc, float* means, float* priors, uchar* data, int numData, int featdim, int numClusters, int flag_norm = 1, int flag_square = 1, int flag_fast = 0);

	int restart_empty_modes(float* posteriors, float* priors, float* means, uchar *data, int numData, int numClusters, int featdim);
	}
};
