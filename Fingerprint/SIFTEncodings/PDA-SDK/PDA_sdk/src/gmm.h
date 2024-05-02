#ifndef GMM_H
#define GMM_H

//std includes
#include <vector>
#include <string>
#include <map>

//vfeat includes
extern "C"
{
	#include <vl/gmm.h>
}
using namespace std;

namespace clustering_and_indexing{

	namespace gmm{

		VlGMM* train_gmm_model(float* desc, int total_data, int dimension, int numclusters, int max_iter, int max_iterkmean);

		void save_gmm_results(char *_path, VlGMM *gmm);

		int save_gmm_results_binary(string folderpath, string exp_name, VlGMM *gmm);

		int save_gmm_results_as_text(string folderpath, string exp_name, VlGMM *gmm);

		int load_gmm_results_binary(string folderpath, string exp_name, float *&means, float *&sigmas, float *&weights, int &K, int &dimension);

		int load_gmm_results_binary(string means_path, string sigmas_path, string weights_path, float *&means, float *&sigmas, float *&weights, int &K, int &dimension);

		void load_gmm_data(string folder_path, float *&means, float *&sigmas, float *&weights, int &K, int &dimension);

	}
};

#endif
