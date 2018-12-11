#ifndef _CUDADP_H_

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <bitset>
#include <iostream>
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
void cudadp_user_kernel(int level, int problem_size, int *deps, void *data);


__global__
void cudadp_kernel(int level, int problem_size, int *deps, void* data) {
    cudadp_user_kernel(level, problem_size, deps, data);
}

__inline__
int compute_blocks(int subproblems, int threads) {
    int blocks = subproblems / threads;
    blocks = subproblems % threads == 0 ? blocks : blocks+1;
    return blocks;
}

__inline__
int compute_subproblems(int m, int n, int level) {
    int subproblems;
    int total_levels = m + n - 1;
    if(level < min(m, n) ) {
        subproblems = level + 1;
    } else if( (total_levels-level) < min(m, n)) {
        subproblems = total_levels - level;
    } else {
        subproblems = min(m, n);
    }
    return subproblems;
}



void cudadp_start(int m, int n, int dep_level, void *data) {
    // generate dependencies;
    int max_levels = m + n - 1;
    cout<<"total levels:"<<max_levels<<endl;

    int *deps;
    int min_mn = min(m, n);
    int blocks = compute_blocks(min_mn, THREADS/2);
    cudaMalloc(&deps, dep_level*min_mn*sizeof(int));

    for(int level = 0; level < max_levels; level+=(THREADS/2+1)) {
        int subproblems = compute_subproblems(m, n, level);
        //cout<<"level:"<<level<<", subproblem size: "<<subproblems<<endl;
        cudadp_kernel<<<compute_blocks(subproblems, THREADS/2), THREADS>>>(level, subproblems, deps, data);
        //cudaDeviceSynchronize();
        //cudaErrorCheck();
    }
    //check_result(deps, dep_level*min_mn);

    cudaFree(deps);
}

#endif
