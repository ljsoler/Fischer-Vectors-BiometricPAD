#include "common.h"
#include <vector>

using namespace std;
using namespace cv;

vector<float> compute_denseSIFT_for_image(const Mat image);

vector<float> compute_denseSIFT_for_image(const Mat image, vector<float>& frames_denseSift);