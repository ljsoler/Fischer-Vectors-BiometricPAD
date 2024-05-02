#include "histogram.h"

#define EPSILON 1.19209290E-07F
#define contrastthreshold 0.0050000001

using namespace std;

namespace histogram
{
	inline  void normalise_histogram(float* hist, int nBins)
	{
		float  norm = 0.0F;

		for (int i = 0; i < nBins; i++)
			norm += (hist[i] * hist[i]);

		norm = sqrtf(norm) + EPSILON;

		for (int i = 0; i < nBins; i++)
			hist[i] /= norm;
	}

	inline bool check_contrains(int x, int y, int width, int height, float paddingX, float paddingY)
	{
		return x - paddingX >= 0 && x + paddingX < width && y - paddingY >= 0 && y + paddingY < height;
	}

	float* calcSpatialHistogram(unsigned int* img, int width, int height, int nBins, int binSize, double bounds[], Geometry geom, int x_step, int y_step, bool norm, bool floatDescriptors, int* numKeypoint)
	{
		int M = height;
		int N = width;

		int frameSizeX = geom.binSizeX * (geom.numBinX - 1) + 1;
		int frameSizeY = geom.binSizeY * (geom.numBinY - 1) + 1;

		float deltaCenterX = 0.5F * geom.binSizeX * (geom.numBinX - 1);
		float deltaCenterY = 0.5F * geom.binSizeY * (geom.numBinY - 1);

		float normConstant = frameSizeX * frameSizeY;

		int boundMinY = MAX(bounds[1], 0);
		int boundMaxY = MIN(bounds[3], M - 1);
		int boundMinX = MAX(bounds[0], 0);
		int boundMaxX = MIN(bounds[2], N - 1);

		int x1 = boundMinX;
		int x2 = boundMaxX;
		int y1 = boundMinY;
		int y2 = boundMaxY;

		int rangeX = x2 - x1 - (geom.numBinX - 1) * geom.binSizeX;
		int rangeY = y2 - y1 - (geom.numBinY - 1) * geom.binSizeY;

		int numFramesX = (rangeX >= 0) ? rangeX / x_step + 1 : 0;
		int numFramesY = (rangeY >= 0) ? rangeY / y_step + 1 : 0;

		int numFrames = numFramesX * numFramesY;

		float* descResult = (float*)malloc(sizeof(float)*numFrames*nBins);
		int idx = 0;

		for (int framey = boundMinY; framey <= boundMaxY - frameSizeY + 1; framey += y_step) {
			for (int framex = boundMinX; framex <= boundMaxX - frameSizeX + 1; framex += x_step)
			{

				int x = framex + deltaCenterX;
				int y = framey + deltaCenterY;

				//Computing histogram for a window size of frameSizeX * frameSizeY
				float* hist = static_cast<float*>(calloc(nBins, sizeof(float)));

				for (int i = 0; i < frameSizeY; i++)
				{
					for (int j = 0; j < frameSizeX; j++)
					{
						hist[img[(i + framey)*N + (j + framex)] / binSize]++;
					}
				}

				/*int mass = 0;
				for (int i = 0; i < nBins; i++)
					mass += hist[i];

				mass /= normConstant;*/

				if (norm)
				{
					//L2-Histogram normalising 
					normalise_histogram(hist, nBins);

					/*for (int i = 0; i < nBins; i++)
						if (hist[i] > 0.2F)
							hist[i] = 0.2F;

					normalise_histogram(hist, nBins);*/
				}

				/*if (floatDescriptors)
				{
					for (int i = 0; i < nBins; ++i)
						hist[i] = mass >= contrastthreshold ? MIN(512.0F * hist[i], 255.0F) : 0;
				}
				else
				{
					for (int i = 0; i < nBins; ++i)
						hist[i] = mass >= contrastthreshold ? ROUND(MIN(512.0F * hist[i], 255.0F)) : 0;
				}*/

				memcpy(&descResult[idx], hist, nBins*sizeof(float));
				idx += nBins;

				free(hist);
			}
		}

		*numKeypoint = idx / nBins;

		return descResult;
	}

	float* calcSpatialHistogramForFixedPoints(unsigned int* img, int width, int height, int nBins, int binSize, Geometry geom, bool norm, bool floatDescriptors, vector<tuple<int, int>> points, int* numKeypoint, int* result_points)
	{
		int M = height;
		int N = width;

		int frameSizeX = geom.binSizeX * (geom.numBinX - 1) + 1;
		int frameSizeY = geom.binSizeY * (geom.numBinY - 1) + 1;

		float deltaCenterX = 0.5F * geom.binSizeX * (geom.numBinX - 1);
		float deltaCenterY = 0.5F * geom.binSizeY * (geom.numBinY - 1);

		float normConstant = frameSizeX * frameSizeY;

		vector<float> desc;


		for (int i = 0; i < points.size(); ++i) {

				int x = get<0>(points[i]);
				int y = get<1>(points[i]);

				if(check_contrains(x, y, N, M, deltaCenterX, deltaCenterY))
				{

					//Computing histogram for a window size of frameSizeX * frameSizeY
					float* hist = static_cast<float*>(calloc(nBins, sizeof(float)));

					for (int j = y - deltaCenterY; j <= y + deltaCenterY; ++j)
					{
						for (int k = x - deltaCenterX; k <= x + deltaCenterX; ++k)
						{
							hist[img[j*N + k] / binSize]++;
						}
					}

					if (norm)
					{
						//L2-Histogram normalising
						normalise_histogram(hist, nBins);
					}

					desc.insert(desc.end(), hist, hist + nBins);
					result_points[i] = 1;



					free(hist);
				}
		}

		float* descResult = (float*)malloc(sizeof(float)*desc.size());
		memcpy(descResult, desc.data(), desc.size()*sizeof(float));

		*numKeypoint = desc.size()/nBins;

		return descResult;
	}

}
