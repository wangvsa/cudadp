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

#define M 100000
#define N 100000
#define G (1000*1000*1000)

struct Sequences {
    char *dev_A;
    char *dev_B;
};


__inline__ __device__
int3 cudadp_user_kernel(int i, int j, int3 left, int3 up, int3 diag, void* data) {

    struct Sequences* seq = (struct Sequences*)data;
    char *A = seq->dev_A;
    char *B = seq->dev_B;
    
    int3 result;
    result.x = max(left.x-Gext, left.z-Gopen);              // E[i,j]
    result.y = max(up.y-Gext, up.z-Gopen);                  // F[i,j]
    result.z = max(0, diag.z + (A[i]==B[j]?MATCH:MISMATCH));  // H[i,j]
    result.z = max3(result.z, result.x, result.y);          // H[i,j]

    return result;
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


    DP_DiagUpLeft sw(40, 30);
    cudadp_start(&sw, dev_seq);

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
