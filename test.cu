#include <iostream>
#include "dp_diag_up_left.h"
using namespace std;

void test(DP *dp) {
    printf("pattern: %d\n", dp->dp_pattern);
    DP_DiagUpLeft *sw = (DP_DiagUpLeft *)dp;
    printf("%d %d %d\n", sw->m, sw->n, dp->total_stages);
}
int main() {
    DP_DiagUpLeft sw(5, 6);
    test(&sw);

    return 0;
}

