


    // Smith-Waterman algorithm with affine gap model
    // MATCH: 1; MISMATCH: -3; Gopen: -3; Gext: -2
    __inline__ __device__
    int3 cudadp_user_kernel(int i, int j, int level,
    						int3 left, int3 up, int3 diag, void *data) {

        struct Sequences* seq = (struct Sequences*)data;
        char *A = seq->dev_A;
        char *B = seq->dev_B;

        int3 result;
        result.x = max(left.x-Gext, left.z-Gopen);                                  // E[i,j]
        result.y = max(up.y-Gext, up.z-Gopen);                                      // F[i,j]
        result.z = max(0, result,x, result.y diag.z + (A[i]==B[j]?MATCH:MISMATCH)); // H[i,j]

        return result;
    }

    // Longest common subsequence
    __inline__ __device__
    int cudadp_user_kernel(int i, in j, int level,
    					   int left, int up, int diag, void *data) {
        struct Sequences* seq = (struct Sequences*)data;
        char *A = seq->dev_A;
        char *B = seq->dev_B;

        int result;
        result = A[i] == B[j] ? 1 : max(left, up);
        return result;
    }



	__inline__ __device__
	int3 cudadp_user_kernel(int m, int n, int level, int3 *deps, void *data) {
	    int tid = blockIdx.x * THREADS + threadIdx.x;

	    struct Sequences* seq = (struct Sequences*)data;
	    char *A = seq->dev_A;
	    char *B = seq->dev_B;

	    int3 *dep1  = deps;
	    int3 *dep2 = &deps[min(m, n)];

	    int i = compute_i(m, n, level);
	    int j = compute_j(m, n, level);

	    // read dependencies from global memory to shared memory
	    __shared__ int3 local_dep1[THREADS+2];
	    __shared__ int3 local_dep2[THREADS+2];
	    if(tid < min(m, n)) {
	        local_dep1[threadIdx.x+1] = dep1[tid];
	        local_dep2[threadIdx.x+1] = dep2[tid];
	    }
	    if(threadIdx.x == THREADS-1 && tid < min(M, N) ) {
	        local_dep1[threadIdx.x+2] = dep1[tid+1];
	        local_dep2[threadIdx.x+2] = dep2[tid+1];
	    }
	    __syncthreads();

	    int3 diag, left, up, result;
	    if (level <= min(M-1, N-1)) {           // up, depends on tid-1, tid
	        left = local_dep2[threadIdx.x];
	        up  = local_dep2[threadIdx.x+1];
	        diag = local_dep1[threadIdx.x];
	    } else {                                // middle and bottom, depends on tid, tid+1
	        left = local_dep2[threadIdx.x+1];
	        up  = local_dep2[threadIdx.x+2];
	        diag = local_dep1[threadIdx.x+2];
	    }

            result.x = max(left.x-Gext, left.z-Gopen);              // E[i,j]
            result.y = max(up.y-Gext, up.z-Gopen);                  // F[i,j]
            result.z = max(0, result.x, result.y, diag.z + (A[i]==Bj?MATCH:MISMATCH));  // H[i,j]
	    return result;
	}

