#include "Vlad_encoding.h"
#include "common.h"

//vfeat includes
extern "C"
{
	#include <vl/vlad.h>
	#include <vl/gmm.h>	
	
}
#include <math.h>

namespace visual_features{

	namespace vlad {

		float* vlad_encoding(VlKDForest* forest, float* centers, float* dataToEncode, int numDataToEncode, int dimension, int numClusters)
		{			
			// find nearest cliuster centers for the data that should be encoded
			vl_uint32* indexes = (vl_uint32*)vl_malloc(sizeof(vl_uint32) * numDataToEncode);
			float* distance = (float*)vl_malloc(sizeof(float)*numDataToEncode);
			vl_kdforest_set_max_num_comparisons(forest, 15);			
			vl_kdforest_query_with_array(forest, indexes, 1, numDataToEncode, distance, dataToEncode);
			// convert indexes array to assignments array,
			// which can be processed by vl_vlad_encode
			float* assignments = (float*)vl_malloc(sizeof(float) * numDataToEncode * numClusters);
			memset(assignments, 0, sizeof(float) * numDataToEncode * numClusters);
			for(int i = 0; i < numDataToEncode; i++) 
			  assignments[i * numClusters + indexes[i]] = 1.;
			vl_free(indexes);
			vl_free(distance);
			// allocate space for vlad encoding
			float* enc = (float*)vl_malloc(sizeof(VL_TYPE_FLOAT) * dimension * numClusters);
			// do the encoding job
			vl_vlad_encode (enc, VL_TYPE_FLOAT,
							centers, dimension, numClusters,
							dataToEncode, numDataToEncode,
							assignments,
							VL_VLAD_FLAG_SQUARE_ROOT | VL_VLAD_FLAG_NORMALIZE_COMPONENTS);		

			vl_free(assignments);

			return enc;
		}
		
	}
}