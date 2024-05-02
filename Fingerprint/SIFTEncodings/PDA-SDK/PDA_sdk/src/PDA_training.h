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

namespace classifier_training
{
	void training_fv_encoder(extract_params params, dsift_extraction_params dsift_params);

	void training_vlad_encoder(extract_params params, dsift_extraction_params dsift_params);

	void training_bovw_encoder(extract_params params, dsift_extraction_params dsift_params);
}
