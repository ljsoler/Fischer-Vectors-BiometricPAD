#ifndef PCA_H_
#define PCA_H_

//#include "TDimensionReductor.h"

#include <opencv/highgui.h>
#include <string>

namespace dimension_reduction{	

	namespace pca{

#define whiteningRegul 0

		int load_model(std::string path, float** means, float** eigenvectors, float** eigenvalues, int* max_components, int* dimension);

		int save_model(std::string path, float* means, float* eigenvalues, float* eigenvectors, int max_components, int dimension);

		float* project(float* descriptors, int numOfData, int dimension, float* means, float* eigenvectors, int max_components);

		void train_model(float* descriptors, int numOfData, int dimension, int max_components, bool whitening, float** means, float** eigenvectors, float** eigenvalues);
	}
}

#endif /* PCA_H_ */
