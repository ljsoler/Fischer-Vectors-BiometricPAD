#ifndef FISHER_ENCODE_H
#define FISHER_ENCODE_H

extern "C"
{
	#include <vl/generic.h>
}

namespace visual_features{

	namespace fisher {

		float* fisher_encoding(float* means, float* sigmas, float* weigths, float* dataToEncode, int numDataToEncode, int dimension, int numClusters);

		float* calculate_sqrt_inverse(vl_size dimension, vl_size numClusters, const float* covariances);

		void precompute_data_for_incremental(vl_size dimension, vl_size numClusters, const float *covariances, const float *priors, float *sqrtInvSigma, float *logCovariances, float *invCovariances, float *logWeights);

		int fisher_encoding_incremental(float* enc, float* inverse, float* means, float* sigmas, float* weigths, float* dataToEncode, int numDataToEncode, int dimension, int numClusters, int totaldata);

		int fisher_encoding_incremental(float* enc, const float *sqrtInvSigma, const float *logCovariances, const float *invCovariances, const float *logWeights, float* means, float* sigmas, float* weigths, float* dataToEncode, int numDataToEncode, int dimension, int numClusters, int totaldata);
	}
}

#endif
