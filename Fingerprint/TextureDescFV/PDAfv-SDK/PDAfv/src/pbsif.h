#include "common.h"
#include <vector>

using namespace std;
using namespace cv;

vector<float> compute_bsif_for_image(const Mat image);

vector<float> compute_multiscale_bsif_for_image(const Mat image);

vector<unsigned char> compute_multiscale_binary_bsif_for_image(const Mat image);
