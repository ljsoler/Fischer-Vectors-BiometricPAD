#ifndef KMEANS_H
#define KMEANS_H

//std includes
#include <vector>
#include <string>
#include <map>

//vfeat includes
extern "C"
{
	#include <vl/gmm.h>
	#include <vl/kdtree.h>
}
using namespace std;

namespace clustering_and_indexing{

	namespace kmeans{

		VlKMeans* train_kmeans_model(float* desc, int total_data, int dimension, int numclusters, int max_iterkmean = 100);

		VlKDForest* build_kdtree(float* vocab, int num_centers, int dimension, int num_tree);
		
		int save_kmeans_results_binary(string path, VlKMeans * kmeans);
		
		int load_kmeans_results_binary(string path, float** vocab, int &dimension, int &num_center);
	}
};

#endif