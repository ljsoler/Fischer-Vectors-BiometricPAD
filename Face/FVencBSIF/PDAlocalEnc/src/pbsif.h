#include "common.h"
#include <vector>

using namespace std;

int read_filter(int filter_dims[], float** filter_data, const char* filter_path);

vector<float> compute_bsif_for_image(const cv::Mat image, char* filter_path);

vector<float> compute_bsif_for_image(const cv::Mat image, int filter_dim[], float* filter_data);

vector<float> compute_bsif_for_image(const cv::Mat image, int filter_dim[], float* filter_data, vector<tuple<int, int>> points);

//vector<float> compute_multiscale_bsif_for_image(const cv::Mat image);

//vector<unsigned char> compute_multiscale_binary_bsif_for_image(const cv::Mat image);
