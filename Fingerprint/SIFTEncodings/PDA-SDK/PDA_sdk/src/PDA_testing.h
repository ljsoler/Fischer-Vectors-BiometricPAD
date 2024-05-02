#pragma once
#include <stdio.h>
#include "common.h"
#include <string>
#include "pca.h"
#include "gmm.h"
#include "phow_dsift.h"
#include <string>

extern "C"
{
	#include <vl/gmm.h>
}

using namespace std;

namespace classifier_testing
{
	void PDAfV_Classify(extract_params params, dsift_extraction_params dsift_params);

	void PDAVlad_Classify(extract_params params, dsift_extraction_params dsift_params);

	void PDABoW_Classify(extract_params params, dsift_extraction_params dsift_params);
}