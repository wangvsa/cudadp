#ifndef _CUDADP_H_

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include "dp_diag_up_left.h"
using namespace std;

#define THREADS 256
#define MAX_DEP 3

#define cudaErrorCheck() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define max3(a,b,c) (max(max(a,b), c))


// User need to implement this function
__inline__ __device__
int3 cudadp_user_kernel(int i, int j, int3 left, int3 up, int3 diag, void *data);


__global__
void cudadp_kernel(DP *dp, int stage, int3 *deps, void* data) {
    int2 coordinates = dp->get_coordinates(threadIdx.x+blockIdx.x*THREADS, stage);
    int i = coordinates.x;
    int j = coordinates.y;
    int3 left = deps[0];
    int3 up = deps[1];
    int3 diag = deps[2];
    cudadp_user_kernel(i, j, left, up, diag, data);
}

// TODO Replace with ceil function in math.h
__inline__
int compute_blocks(int subproblems, int threads) {
    int blocks = subproblems / threads;
    blocks = subproblems % threads == 0 ? blocks : blocks+1;
    return blocks;
}


void check_result(int3 *result, int length) {
    int3 h_result[length];

    cudaMemcpy(h_result, result, length*sizeof(int3), cudaMemcpyDeviceToHost);
    for(int i = 0; i < length; i++) {
        printf("%d %d %d ", h_result[i].x, h_result[i].y, h_result[i].z);
    }
    printf("\n");
}


void cudadp_start(DP *dp, void* data) {

    int dep_level = 2;
    printf("m: %d, n: %d, stages: %d\n", dp->m, dp->n, dp->total_stages);

    int3 *deps;
    int min_mn = min(dp->m, dp->n);
    int blocks = compute_blocks(min_mn, THREADS/2);
    cudaMalloc(&deps, dep_level*min_mn*sizeof(int3));

    for(int level = 0; level < dp->total_stages; level+=(THREADS/2+1)) {
        int subproblems = dp->get_problem_size(level);
        //cout<<"level:"<<level<<", subproblem size: "<<subproblems<<endl;
        cudadp_kernel<<<compute_blocks(subproblems, THREADS/2), THREADS>>>(dp, level, deps, data);
        //cudaDeviceSynchronize();
        //cudaErrorCheck();
    }
    //check_result(deps, dep_level*min_mn);

    cudaFree(deps);
}

#endif
