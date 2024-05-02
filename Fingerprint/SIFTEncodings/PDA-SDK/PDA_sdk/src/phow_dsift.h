#include "common.h"
#include <vector>

using namespace std;
using namespace cv;



vector<float> compute_for_image(const Mat image, dsift_extraction_params params);

vector<float> compute_for_image(const Mat image, dsift_extraction_params params, vector<float>& frames_denseSift);