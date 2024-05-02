#include "Bovw_encoding.h"
#include "common.h"

//vfeat includes
extern "C"
{
	#include <vl/vlad.h>
	#include <vl/gmm.h>	
	#include <vl/homkermap.h>
	
}

using namespace std;

namespace visual_features{

	namespace bovw {

#define SpatialLayout_size 2
		unsigned char numSpatialX[SpatialLayout_size] = {2, 4};
		unsigned char numSpatialY[SpatialLayout_size] = {2, 4};
		
		float* bovw_encoding(VlKDForest* forest, float* centers, float* dataToEncode, int numDataToEncode, int dimension, int numClusters, int w, int h, float* frames)
		{		
			// find nearest cliuster centers for the data that should be encoded
			vl_uint32* binsa = (vl_uint32*)vl_malloc(sizeof(vl_uint32) * numDataToEncode);
			float* distance = (float*)vl_malloc(sizeof(float)*numDataToEncode);
			vl_kdforest_set_max_num_comparisons(forest, 50);			
			vl_kdforest_query_with_array(forest, binsa, 1, numDataToEncode, distance, dataToEncode);		

			vector<float> bovw_enc;

			for (int i = 0; i < SpatialLayout_size; ++i)
			{
				int count_binsX = numSpatialX[i] + 1;
				int count_binsY = numSpatialY[i] + 1;

				unsigned int* Bx = (unsigned int*)malloc(sizeof(unsigned int)*count_binsX);
				unsigned int* By = (unsigned int*)malloc(sizeof(unsigned int)*count_binsY);
				
				int bin_sizeX = w/(count_binsX - 1);
				int bin_sizeY = h/(count_binsY - 1);

				for (int j = 0; j < count_binsX; j++)
				{
					Bx[j] = j*bin_sizeX;
					By[j] = j*bin_sizeY;
				}

				int* idx = (int*)malloc(sizeof(int)*numDataToEncode);
				int* IDY = (int*)malloc(sizeof(int)*numDataToEncode);
				for (int j = 0 ; j < numDataToEncode ; ++j) 
				{
					  double x = frames[j*2] ;
					  vl_uindex blower = 0 ;
					  vl_uindex bupper = count_binsX - 1 ;
					  vl_uindex bsplit ;

					  if (x < Bx[0]) {
						idx[j] = 0 ;
						continue ;
					  }

					  if (x >= Bx[count_binsX - 1]) {
						  idx[j] = count_binsX ;
						continue ;
					  }

					  while (blower + 1 < bupper) {
						bsplit = (bupper + blower) / 2 ;
						if (x < Bx[bsplit]) 
							bupper = bsplit ;
						else
							blower = bsplit ;
					  }
					  idx[j] = blower;

					  double x_y = frames[j*2 + 1] ;
					  vl_uindex blowerY = 0 ;
					  vl_uindex bupperY = count_binsY - 1 ;
					  vl_uindex bsplitY ;

					  if (x_y < By[0]) {
						IDY[j] = 0 ;
						continue ;
					  }

					  if (x_y >= By[count_binsY - 1]) {
						  IDY[j] = count_binsY ;
						continue ;
					  }

					  while (blowerY + 1 < bupperY) {
						bsplitY = (bupperY + blowerY) / 2 ;
						if (x_y < By[bsplitY]) 
							bupperY = bsplitY ;
						else
							blowerY = bsplitY ;
					  }
					  IDY[j] = blowerY;
				}		
						
				free(Bx);
				free(By);
				//sub2ind	
				int* binsX = idx;
				int* binsY = IDY;
				unsigned int* bins = (unsigned int*)malloc(sizeof(unsigned int)*numDataToEncode);
				int w = numSpatialY[i], h = numSpatialX[i];
				for (int j = 0; j < numDataToEncode; j++)
					bins[j] = binsa[j]*w*h + binsX[j]*h + binsY[j];

				float* hist = (float*)calloc(numSpatialY[i]*numSpatialX[i]*numClusters, sizeof(float));

				for (int j = 0; j < numDataToEncode; j++)
					hist[bins[j]] += 1.0;

				float suma = 0;
				for (int j = 0; j < numSpatialY[i]*numSpatialX[i]*numClusters; j++)
					suma += hist[j];
				for (int j = 0; j < numSpatialY[i]*numSpatialX[i]*numClusters; j++)
					 hist[j] /= suma;
				bovw_enc.insert(bovw_enc.end(), hist, hist + numSpatialY[i]*numSpatialX[i]*numClusters);
				free(binsX);
				free(binsY);
				free(bins);
				free(hist);
			}

			// allocate space for bovw encoding
			float suma = 0;
			for (int i = 0; i < bovw_enc.size(); i++)
				suma += bovw_enc[i];
			for (int i = 0; i < bovw_enc.size(); i++)
				bovw_enc[i] /= suma;
			double gamma = 0.5 ;
			int order = 1 ;
			double period = -1 ; // use default
			float* enc_features = (float*)vl_malloc(sizeof(VL_TYPE_FLOAT) * 3 * bovw_enc.size());
			vl_size psiStride = 1 ;
			double x = 0.5 ;
			VlHomogeneousKernelMap * hom = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, gamma, order, period, VlHomogeneousKernelMapWindowRectangular) ;	

			for (int i = 0; i < bovw_enc.size(); i++)
				vl_homogeneouskernelmap_evaluate_f(hom, &enc_features[3*i], psiStride, bovw_enc[i]);							
			
			vl_homogeneouskernelmap_delete(hom);		

			vl_free(binsa);
			vl_free(distance);

			return enc_features;
		}		
	}
}