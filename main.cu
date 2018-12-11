#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <bitset>
#include <iostream>
#include "cudadp.h"
#include "fasta_util.h"
using namespace std;

// Affine gap model
#define MATCH 1
#define MISMATCH 1
#define Gopen -3
#define Gext -2

#define M 1000000
#define N 100000
#define G (1024*1024*1024)

struct Sequences {
    char *dev_A;
    char *dev_B;
};

__inline__ __device__
int compute_j(int level, int problem_size) {
    int tid = blockIdx.x * THREADS + threadIdx.x;
    int j;
    if (level <= min(M-1, N-1)) {           // up
        j = tid;
    } else if(level > max(M-1, N-1)) {      // bottom
        j = N - problem_size + tid;
    } else {                                // middle
        j = level - min(M-1, N-1) + tid;
    }
    return j;
}

__inline__ __device__
void cudadp_user_kernel(int level, int problem_size, int3 *deps, void *data) {
    int tid = blockIdx.x * THREADS + threadIdx.x;
    //if(tid >= problem_size) return;

    struct Sequences* seq = (struct Sequences*)data;
    char *A = seq->dev_A;
    char *B = seq->dev_B;
    
    int3 *dep1  = deps;
    int3 *dep2 = &deps[min(M, N)];

    // read dependencies from global memory to shared memory
    __shared__ int3 local_dep1[THREADS+2];
    __shared__ int3 local_dep2[THREADS+2];
    if(tid < min(M, N)) {
        local_dep1[threadIdx.x+1] = dep1[tid];
        local_dep2[threadIdx.x+1] = dep2[tid];
    }
    if(threadIdx.x == THREADS-1 && tid < min(M, N) ) {
        local_dep1[threadIdx.x+2] = dep1[tid+1];
        local_dep2[threadIdx.x+2] = dep2[tid+1];
    }
    __syncthreads();

    int i, j;
    j = compute_j(level, problem_size);
    i = level - j;
    char Bj = B[j]; // j, B[j] are not changing during following steps

    int3 diag, left, up, result;
    for(int k = 0; k < THREADS/2+1; k++, i++, level++) {

        if(level >= M+N-1) return;

        if(threadIdx.x>=THREADS-k) {
            //printf("level:%d (%d, %d)\n", level, i, j);

            if (level <= min(M-1, N-1)) {           // up, depends on tid-1, tid
                left = local_dep2[threadIdx.x];
                //up  = local_dep2[threadIdx.x+1];
                up = result;
                diag = local_dep1[threadIdx.x];
            } else {                                // middle and bottom, depends on tid, tid+1
                //left = local_dep2[threadIdx.x+1];
                left = result;
                up  = local_dep2[threadIdx.x+2];
                diag = local_dep1[threadIdx.x+2];
            }

            result.x = max(left.x-Gext, left.z-Gopen);              // E[i,j]
            result.y = max(up.y-Gext, up.z-Gopen);                  // F[i,j]
            result.z = max(0, diag.z + (A[i]==Bj?MATCH:MISMATCH));  // H[i,j]
            result.z = max3(result.z, result.x, result.y);          // H[i,j]

            if(k == THREADS/2 || level==M+N-2) {                    // last level, write into global memory
                deps[tid] = result;
            } else {                                                // intermediate levels, use local memory
                // swap dependency levels
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



string random_string(int length) {
    //srand (time(0) );
    string s(length, 'A');
    const char alphabet[] = {'A', 'C', 'G', 'T'};
    for(int i = 0; i < length; i++) {
        s[i] = alphabet[(rand() % 4)];
    }
    return s;
}


int main(int argc, char *argv[]) {
    //string A = read_fasta_file(argv[1]);
    //string B = read_fasta_file(argv[1]);
    //string A = "GTCTTACATCCGTTCG";
    //string B = "GTCTTACATCCGTTCG";
    string A = random_string(M);
    string B = random_string(N);
    //printf("A:%s\nB:%s\n", A.c_str(), B.c_str());

    struct Sequences seq;
    cudaMalloc(&(seq.dev_A), sizeof(char) * A.length());
    cudaMalloc(&(seq.dev_B), sizeof(char) * B.length());
    cudaMemcpy(seq.dev_A, A.c_str(), sizeof(char)*A.length(), cudaMemcpyHostToDevice);
    cudaMemcpy(seq.dev_B, B.c_str(), sizeof(char)*B.length(), cudaMemcpyHostToDevice);

    struct Sequences *dev_seq;
    cudaMalloc(&dev_seq, sizeof(struct Sequences));
    cudaMemcpy(dev_seq, &seq, sizeof(struct Sequences), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudadp_start(A.length(), B.length(), 2, dev_seq);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float gcpus = A.length() * B.length() * 1.0 / G / (milliseconds/1000.0);
    printf("time:%f, GCPUS: %f\n", milliseconds/1000.0, gcpus);


    cudaFree(seq.dev_A);
    cudaFree(seq.dev_B);
    cudaFree(dev_seq);

    return 0;
}
