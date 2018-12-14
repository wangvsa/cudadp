#ifndef _DP_H_

/*
 * Abstract Class
 *
 */
class DP {
    int m;      // rows
    int n;      // columns
    DP(int m_, int n_) {
        m = m_;
        n = n_;
    }

    virtual int2 get_coordinates(int tid, int level) = 0;

    virtual int get_problem_size(int level) = 0;
};



#endif
