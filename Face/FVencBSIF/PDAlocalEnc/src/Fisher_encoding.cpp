#include "Fisher_encoding.h"
//vfeat includes
extern "C"
{
	#include <vl/fisher.h>
	#include <vl/gmm.h>
}
#include <math.h>

namespace visual_features{

	namespace fisher {

#define VL_GMM_MIN_PRIOR 1e-6

		float* fisher_encoding(float* means, float* sigmas, float* weigths, float* dataToEncode, int numDataToEncode, int dimension, int numClusters)
		{
			// allocate space for the encoding
			float* enc = (float*)vl_malloc(sizeof(float) * 2 * dimension * numClusters);

			// run fisher encoding
			vl_fisher_encode(enc, VL_TYPE_FLOAT, means, dimension, numClusters,
				sigmas, weigths, dataToEncode, numDataToEncode,
				VL_FISHER_FLAG_IMPROVED);

			return enc;
		}

		float* calculate_sqrt_inverse(vl_size dimension, vl_size numClusters, const float* covariances)
		{
			assert(numClusters >= 1);
			assert(dimension >= 1);

			float * sqrtInvSigma = (float*)vl_malloc(sizeof(float) * dimension * numClusters);

			vl_index i_cl, i_d;
			for (i_cl = 0; i_cl < (signed)numClusters; ++i_cl) {
				for (i_d = 0; i_d < dimension; i_d++) {
					sqrtInvSigma[i_cl*dimension + i_d] = sqrt(1.0 / covariances[i_cl*dimension + i_d]);
				}
			}
			return sqrtInvSigma;
		}

		void precompute_data_for_incremental(vl_size dimension, vl_size numClusters, const float *covariances, const float *priors, float *sqrtInvSigma, float *logCovariances, float *invCovariances, float *logWeights)
		{
			vl_index i_cl, i_d;

			sqrtInvSigma =		(float *)vl_malloc(sizeof(float) * numClusters * dimension);
			logCovariances =	(float *)vl_malloc(sizeof(float) * numClusters);
			invCovariances =	(float *)vl_malloc(sizeof(float) * numClusters * dimension);
			logWeights =		(float *)vl_malloc(sizeof(float) * numClusters);

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) num_threads(vl_get_max_threads())
#endif
			for (i_cl = 0; i_cl < (signed)numClusters; ++i_cl) {
				float logSigma = 0;
				
				if (priors[i_cl] < VL_GMM_MIN_PRIOR) 
					logWeights[i_cl] = -(float)VL_INFINITY_D;
				else 
					logWeights[i_cl] = log(priors[i_cl]);
				
				for (i_d = 0; i_d < dimension; ++i_d) {
					logSigma += log(covariances[i_cl*dimension + i_d]);
					invCovariances[i_cl*dimension + i_d] = (float) 1.0 / covariances[i_cl*dimension + i_d];
					sqrtInvSigma[i_cl*dimension + i_d] = sqrt(1.0 / covariances[i_cl*dimension + i_d]);
				}
				logCovariances[i_cl] = logSigma;
			}
		}

		/*int fisher_encoding_incremental(float* enc, float* inverse, float* means, float* sigmas, float* weigths, float* dataToEncode, int numDataToEncode, int dimension, int numClusters, int totaldata)
		{
			my_vl_fisher_encode_incremental(enc, VL_TYPE_FLOAT, inverse, means, dimension, numClusters, sigmas, weigths, dataToEncode, numDataToEncode, totaldata, VL_FISHER_FLAG_IMPROVED | VL_FISHER_FLAG_FAST);
			return 1;
		}

		int fisher_encoding_incremental(float* enc, const float *sqrtInvSigma, const float *logCovariances, const float *invCovariances, const float *logWeights, float* means, float* sigmas, float* weigths, float* dataToEncode, int numDataToEncode, int dimension, int numClusters, int totaldata)
		{
			my_vl_fisher_encode_incremental(enc, VL_TYPE_FLOAT, sqrtInvSigma, logCovariances, invCovariances, logWeights, means, dimension, numClusters, sigmas, weigths, dataToEncode, numDataToEncode, totaldata, VL_FISHER_FLAG_IMPROVED | VL_FISHER_FLAG_FAST);
			return 1;
		}*/
	}
}
