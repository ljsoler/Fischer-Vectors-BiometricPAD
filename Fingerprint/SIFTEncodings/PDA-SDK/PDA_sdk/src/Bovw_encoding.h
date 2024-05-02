#ifndef BOVW_ENCODE_H
#define BOVW_ENCODE_H


extern "C"
{
#include <vl/kdtree.h>
}

namespace visual_features{

	namespace bovw {

		float* bovw_encoding(VlKDForest* forest, float* centers, float* dataToEncode, int numDataToEncode, int dimension, int numClusters, int w, int h, float* frames);
	}
}

#endif
