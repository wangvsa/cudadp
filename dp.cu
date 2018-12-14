#ifndef _DP_H_
#include "dp.h"

/**
  * DP Class for Smiwth-Waterman, LCS, etc.
  * each cell relies on three cells: left, up, and diag
  */
class DP_DiagUpLeft : DP {
    int m;      // rows
    int n;      // columns

    virtual int2 get_coordinates(int tid, int level) {
        int2 coordinates;
        int problem_size = get_problem_size(level);

        int j;
        if (level <= min(m-1, n-1)) {           // up
            j = tid;
        } else if(level > max(m-1, n-1)) {      // bottom
            j = n - problem_size + tid;
        } else {                                // middle
            j = level - min(m-1, n-1) + tid;
        }
        coordinates.y = j;                      // j
        coordinates.x = level - j;              // i
        return coordinates;
    }

    virtual int get_problem_size(int level) {
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
};



#endif
