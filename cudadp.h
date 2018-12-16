#ifndef _CUDADP_H_

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include "dp_diag_up_left.h"
using namespace std;

#define THREADS 256

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
void cudadp_kernel(DP *tmp, int stage, int3 *deps, void* data) {
    DP_DiagUpLeft *dp = (DP_DiagUpLeft *)tmp;
    int tid = threadIdx.x + blockIdx.x * THREADS;
    int2 coordinates = dp->get_coordinates(tid, stage);
    int i = coordinates.x;
    int j = coordinates.y;

    //printf("stage:%d (%d, %d) %d\n", stage, i, j, dp->total_stages);
    // read dependencies from global memory to shared memory
    int3 *dep1  = deps;
    int3 *dep2 = &(deps[min(dp->m, dp->n)]);

    __shared__ int3 local_dep1[THREADS+2];
    __shared__ int3 local_dep2[THREADS+2];
    if(tid < min(dp->m, dp->n)) {
        local_dep1[threadIdx.x+1] = dep1[tid];
        local_dep2[threadIdx.x+1] = dep2[tid];
    }
    if(threadIdx.x == THREADS-1 && tid < min(dp->m, dp->n) ) {
        local_dep1[threadIdx.x+2] = dep1[tid+1];
        local_dep2[threadIdx.x+2] = dep2[tid+1];
    }
    __syncthreads();

    int3 diag, left, up, result;
    for(int k = 0; k < THREADS/2+1; k++, i++, stage++) {
        if(stage >= dp->total_stages) return;

        if(threadIdx.x < THREADS-k && i >=0 && i < dp->m && j >= 0 && j < dp->n) {
            //printf("stage:%d (%d, %d) %d\n", stage, i, j, dp->total_stages);
            if (stage <= min(dp->m-1, dp->n-1)) {           // up, depends on tid-1, tid
                left = local_dep2[threadIdx.x];
                up  = local_dep2[threadIdx.x+1];
                diag = local_dep1[threadIdx.x];
            } else {                                // middle and bottom, depends on tid, tid+1
                left = local_dep2[threadIdx.x+1];
                up  = local_dep2[threadIdx.x+2];
                diag = local_dep1[threadIdx.x+2];
            }

            result = cudadp_user_kernel(i, j, left, up, diag, data);
            if(k == THREADS/2 || stage==dp->total_stages-1) {       // last stage, write into global memory
                if(tid < min(dp->m, dp->n))
                    dep2[tid] = result;
            } else {                                                // intermediate stages, use local memory
                // swap dependency stages
                local_dep2[threadIdx.x] = local_dep1[threadIdx.x];
                if(threadIdx.x == 0) {
                    local_dep2[THREADS] = local_dep1[THREADS];
                    local_dep2[THREADS+1] = local_dep1[THREADS+1];
                }
                local_dep1[threadIdx.x+1] = result;
            }

        }
        __syncthreads();
    }
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


void cudadp_start(DP *tmp, void* data) {

    DP_DiagUpLeft *dp = (DP_DiagUpLeft *)tmp;

    int dep_level = 2;

    int3 *deps;
    int min_mn = min(dp->m, dp->n);
    int blocks = compute_blocks(min_mn, THREADS/2);
    cudaMalloc(&deps, dep_level*min_mn*sizeof(int3));

    DP *dev_dp;
    cudaMalloc(&dev_dp, sizeof(DP));
    cudaMemcpy(dev_dp, dp, sizeof(DP), cudaMemcpyHostToDevice);

    for(int stage= 0; stage< dp->total_stages; stage+=(THREADS/2+1)) {
        int subproblems = dp->get_problem_size(stage);
        //cout<<"stage:"<<stage<<", subproblem size: "<<subproblems<<endl;
        cudadp_kernel<<<compute_blocks(subproblems, THREADS/2), THREADS>>>(dev_dp, stage, deps, data);
        cudaDeviceSynchronize();
        cudaErrorCheck();
    }
    //check_result(deps, dep_level*min_mn);

    cudaFree(deps);
    cudaFree(dev_dp);
}

#endif
