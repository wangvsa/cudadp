#ifndef _DP_DIAG_UP_LEFT_H_
#include "dp.h"

/**
  * DP Class for Smiwth-Waterman, LCS, etc.
  * each cell relies on three cells: left, up, and diag
  */
class DP_DiagUpLeft : public DP {
public:
    DP_DiagUpLeft(int m_, int n_) {
        m = m_;
        n = n_;
        total_stages = m + n - 1;
        //dp_pattern = DP_PATTERN_DIAG_UP_LEFT;
    }

    __inline__ __device__
    int2 get_coordinates(int tid, int stage) {
        int2 coordinates;
        int problem_size = get_problem_size(stage);

        int j;
        if (stage <= min(m-1, n-1)) {           // up
            j = tid;
        } else if(stage > max(m-1, n-1)) {      // bottom
            j = n - problem_size + tid;
        } else {                                // middle
            j = stage - min(m-1, n-1) + tid;
        }
        coordinates.y = j;                      // j
        coordinates.x = stage - j;              // i
        return coordinates;
    }

    __inline__ __device__ __host__
    int get_problem_size(int stage) {
        int subproblems;
        if(stage < min(m, n) ) {
            subproblems = stage + 1;
        } else if( (total_stages-stage) < min(m, n)) {
            subproblems = total_stages - stage;
        } else {
            subproblems = min(m, n);
        }
        return subproblems;
    }
};


#endif
