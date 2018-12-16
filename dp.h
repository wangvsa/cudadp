#ifndef _DP_H_


/*
 * Abstract Class
 * virtual functions need to implemented in children classes
 */
class DP {
public:
    int m;                  // rows
    int n;                  // columns
    int total_stages;       // total number of stages

    DP() { }

    DP(int m_, int n_) {
        m = m_;
        n = n_;
        total_stages = 0;
    }

    // subclass needs to override this function
    __inline__ __device__
    int2 get_coordinates(int tid, int level) {
        int2 coordinates = {0, 0};
        return coordinates;
    }

    // subclass needs to override this function
    __inline__ __device__ __host__
    int get_problem_size(int stage) {
        return 0;
    }
};


#endif
