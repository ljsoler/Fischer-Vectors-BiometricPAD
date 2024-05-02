/*
 * random_sampling.c
 *
 *  Created on: 12/04/2017
 *      Author: destevez
 *
 *    void shuffle(int *array, int n):
 *    -implements Fisher–Yates shuffle aka. Knuth shuffle on arbitrary array of integers
 */


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef WIN32
#include <process.h>
#include <sys/stat.h>
#define GETPID _getpid()
#define	RAND_MAX_OWN 2147483647
#else
#define GETPID getpid()
#define RAND_MAX_OWN RAND_MAX
#endif



int rand_int(int n)
{
    int limit = RAND_MAX_OWN - RAND_MAX_OWN % n;
    int rnd;
    do {
        rnd = rand();
    }
    while (rnd >= limit);
    return rnd % n;
}

#ifndef WIN32
	static int rdtsc()
	{
		 __asm__ __volatile__("rdtsc");
	}	
#endif

	void shuffle(int *array, int n)
	{
	    int i, j, tmp;
	    //seed selection
#ifdef WIN32
		  srand(GETPID + time(NULL)); //numero de ciclos + process ID+hora del sistema
#else
		  srand(rdtsc()+ GETPID + (int)time(NULL)); //numero de ciclos + process ID+hora del sistema
#endif	  
	
	    for (i = n - 1; i > 0; i--)
	    {
	        j = rand_int(i + 1);
	        tmp = array[j];
	        array[j] = array[i];
	        array[i] = tmp;
	   }
	}



