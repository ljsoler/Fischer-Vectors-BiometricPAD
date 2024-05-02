//#ifndef VL_FISHER_H
//#define VL_FISHER_H

extern "C"
{
	#include <vl/fisher.h>
	#include <vl/generic.h>
}

namespace visual_features {

	namespace fisher {
		
		vl_size my_vl_fisher_encode
			(void * enc, vl_type dataType,
				void const * means, vl_size dimension, vl_size numClusters,
				void const * covariances,
				void const * priors, const void* data, vl_size numData,	int flags);

		vl_size my_vl_fisher_encode_incremental
			(float * enc,
				vl_type dataType,
				float * sqrtInvSigma,
				float const * means, vl_size dimension, vl_size numClusters,
				float const * covariances,
				float const * priors,
				float const * data, vl_size numData,
				int totaldata,
				int flags);

		vl_size my_vl_fisher_encode_incremental
			(float * enc,
				vl_type dataType,
				const float * sqrtInvSigma,
				const float * logCovariances,
				const float * invCovariances,
				const float * logWeights,
				float const * means, vl_size dimension, vl_size numClusters,
				float const * covariances,
				float const * priors,
				float const * data, vl_size numData,
				int totaldata,
				int flags);
	}
}
/* VL_FISHER_H */
//#endif
