#ifndef GMM_H
#define GMM_H

//std includes
#include <string>
#include <map>

//vfeat includes
extern "C"
{
	#include <vl/gmm.h>
}

namespace clustering_and_indexing{

	namespace gmm{

		VlGMM* train_gmm_model(float* desc, int total_data, int dimension, int numclusters, int max_iter, int max_iterkmean);

		void save_gmm_results(char *_path, VlGMM *gmm);

		int save_gmm_results_binary(std::string folderpath, std::string exp_name, VlGMM *gmm);

		int save_gmm_results_as_text(std::string folderpath, std::string exp_name, VlGMM *gmm);

		int load_gmm_results_binary(std::string folderpath, std::string exp_name, float *&means, float *&sigmas, float *&weights, int &K, int &dimension);

		int load_gmm_results_binary(std::string means_path, std::string sigmas_path, std::string weights_path, float *&means, float *&sigmas, float *&weights, int &K, int &dimension);

		void load_gmm_data(std::string folder_path, float *&means, float *&sigmas, float *&weights, int &K, int &dimension);

	}
};

#endif
