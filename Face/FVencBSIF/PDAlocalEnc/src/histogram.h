#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include "common.h"

namespace histogram
{
	float* calcSpatialHistogram(unsigned int* img, int width, int height, int nBins, int binSize, double bounds[], Geometry geom, int x_step, int y_step, bool norm, bool floatDescriptors, int* numKeypoint);

	float* calcSpatialHistogramForFixedPoints(unsigned int* img, int width, int height, int nBins, int binSize, Geometry geom, bool norm, bool floatDescriptors, std::vector<std::tuple<int, int>> points, int* numKeypoint, int* result_points);
}
#endif
