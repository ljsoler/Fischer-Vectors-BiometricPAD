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

namespace pdafv
{	
	void training_fv_encoder_gmm(extract_params<float> params);
	
	void testing_fv_encoder_gmm(extract_params<float> params);

	void training_fv_encoder_bmm(extract_params<unsigned char> params);

	void testing_fv_encoder_bmm(extract_params<unsigned char> params);
}
