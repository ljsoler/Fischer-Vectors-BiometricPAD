#ifndef VLAD_ENCODE_H
#define VLAD_ENCODE_H

extern "C"
{
	#include <vl/generic.h>
	#include <vl/kdtree.h>
}

namespace visual_features{

	namespace vlad {

		float* vlad_encoding(VlKDForest* forest, float* centers, float* dataToEncode, int numDataToEncode, int dimension, int numClusters);
	}
}

#endif
