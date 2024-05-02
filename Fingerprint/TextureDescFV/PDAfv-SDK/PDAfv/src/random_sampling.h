/*
 * random_sampling.h
 *
 *  Created on: 12/04/2017
 *      Author: destevez
 */


#ifndef RANDOM_SAMPLING_H_
#define RANDOM_SAMPLING_H_

#define rand_select(num_rand,min_value,max_value,value)\
{\
	value = num_rand * (max_value - min_value) / RAND_MAX + min_value;\
}


#ifdef __cplusplus
extern "C" {
#endif

void shuffle(int *array, int n);
int rand_int(int n);

#ifdef __cplusplus
}
#endif

#endif /* RANDOM_SAMPLING_H_ */
