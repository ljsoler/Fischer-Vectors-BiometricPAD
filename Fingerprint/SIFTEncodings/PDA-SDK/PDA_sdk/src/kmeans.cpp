#include "kmeans.h"
#include "common.h"
#ifdef WIN32
#include <process.h>
#endif

#define NUM_TREES 3

#define NUM_REPITATIONS 1

using namespace std;

namespace clustering_and_indexing{

	namespace kmeans{

		VlKMeans* train_kmeans_model(float* desc, int total_data, int dimension, int numclusters, int max_iterkmean)
		{
			// create a GMM object and cluster input data to get means, covariances
			// and priors of the estimated mixture	
			/*double energy;
			double * centers;*/
			float* data = desc;
			VlRand* random = vl_get_rand();
			vl_rand_seed(random, 1);
			// Use float data and the L2 distance for clustering
			VlKMeans* kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
			// Use Elkan algorithm
			vl_kmeans_set_algorithm (kmeans, VlKMeansElkan) ;			
			// Use Plus Plus initialization
			vl_kmeans_set_initialization(kmeans, VlKMeansPlusPlus);
			// Run at most 100 iterations of cluster refinement using Lloyd algorithm
			vl_kmeans_set_max_num_iterations (kmeans, max_iterkmean) ;

			vl_kmeans_set_verbosity(kmeans, 1);

			vl_kmeans_set_num_repetitions(kmeans, NUM_REPITATIONS);
			
			vl_kmeans_set_num_trees(kmeans, NUM_TREES);

			vl_kmeans_set_max_num_comparisons(kmeans, max_iterkmean);			

			vl_kmeans_cluster(kmeans, data, dimension, total_data, numclusters);			
			
			return kmeans;			
		}

		VlKDForest* build_kdtree(float* vocab, int num_centers, int dimension, int num_tree)
		{
			VlRand* random = vl_get_rand();
			vl_rand_seed(random, 1);
			VlKDForest* forest = vl_kdforest_new(VL_TYPE_FLOAT, dimension, num_tree, VlDistanceL2);
			vl_kdforest_set_thresholding_method(forest, VL_KDTREE_MEDIAN);
			vl_kdforest_build (forest, num_centers, vocab);
			return forest;
		}
		
		int save_kmeans_results_binary(string path, VlKMeans * kmeans)
		{
			FILE * ofp;
			const float* centers = (const float*)vl_kmeans_get_centers(kmeans);
			int num_center = vl_kmeans_get_num_centers(kmeans);
			int dimension = vl_kmeans_get_dimension(kmeans);
			//vl_type data_type = vl_kmeans_get_data_type(kmeans);
			
			ofp = fopen(path.c_str(), "wb"); //  save kmeans centers
			if (ofp == NULL)
				return -1;			

			fwrite(&num_center, sizeof(int), 1, ofp);	
			fwrite(&dimension, sizeof(int), 1, ofp);				
			fwrite(centers, sizeof(float), num_center*dimension, ofp);
			
			fclose(ofp);		

			return 1;
		}
		
		int load_kmeans_results_binary(string path, float** vocab, int &dimension, int &num_center)
		{
			FILE* ofp = fopen(path.c_str(), "rb"); //  [means, sigmas, weights]
			if (ofp == NULL)
				return -1;

			vl_type data_type = 0;
			int ncenters = 0, dim = 0;
			fread(&ncenters, sizeof(int), 1, ofp);
			fread(&dim, sizeof(int), 1, ofp);			
			float* centers = (float*)malloc(sizeof(float)*ncenters*dim);
			fread(centers, sizeof(float), ncenters*dim, ofp);
			*vocab = centers;
			num_center = ncenters;
			dimension = dim;			
			fclose(ofp);

			return 1;
		}
	}
};