#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <bitset>
#include <iostream>
#include <vector>
#include <ctime>
using namespace std;

#define G (1000*1000*1000)
#define M 10240*5 // length of A
#define N 10240*10 // length of B

const char alphabet[] = {'A', 'C', 'G', 'T'};

vector<bitset<M> > compute_alphabet_strings(string A) {
    vector<bitset<M> > alphabet_strings;
    for(int i = 0; i < 4; i++) {
        bitset<M> alphabet_string;
        for(int j = 0; j < A.length(); j++) {
            if(alphabet[i] == A[j])
                alphabet_string[M-j-1] = 1;
            else
                alphabet_string[M-j-1] = 0;
        }
        alphabet_strings.push_back(alphabet_string);
    }
    return alphabet_strings;
}

inline
bitset<M> get_alphabet_string(char bi, vector<bitset<M> > alphabet_strings) {
    int k;
    for(int i = 0; i < 4; i++)
        if(bi==alphabet[i])
            k = i;
    return alphabet_strings[k];
}

bitset<M> bits_substraction(bitset<M> a, bitset<M> b) {
    bitset<M> c;
    bool borrow = false;

    for(int i = 0; i < a.size(); i++) {
        if(a[i]==b[i]) {
            c[i] = 0;
            if(borrow)
                c[i] = 1;       // borrow still is true
        } else {
            c[i] = 1;
            if(borrow)
                c[i] = 0;
            if(a[i] == 0)
                borrow = true;
            else
                borrow = false;
        }
    }
    //cout<<a<<" "<<b<<" "<<c<<endl;
    return c;
}

bool is_parallel(bitset<M> a, bitset<M> b) {
    if(a == b)
        return true;
    return false;
}

void compute_matrix(string B, int start_i, int end_i, vector<bitset<M> > rows, vector<bitset<M> > alphabet_strings) {
    if(start_i != 1) rows[start_i-1][5] = 1;
    bitset<M> x;
    bitset<M> tmp;
    for(int i = start_i; i < end_i; i++) {  // i start from 1 so B[i-1]
        x = rows[i-1] | get_alphabet_string(B[i-1], alphabet_strings);
        tmp = rows[i-1] << 1;
        tmp[0] = 1;
        rows[i] = x & (bits_substraction(x, tmp) ^ x);
        //cout<<x<<" "<<bits_substraction(x, tmp)<<" "<<rows[i]<<endl;
    }
}

void compute_matrix(string B, int start_i, int end_i, vector<bitset<M> > alphabet_strings) {
    bitset<M> row;

    bitset<M> x;
    bitset<M> tmp;
    for(int i = start_i; i < end_i; i++) {  // i start from 1 so B[i-1]
        x = row | get_alphabet_string(B[i-1], alphabet_strings);
        tmp = row << 1;
        tmp[0] = 1;
        row = x & (bits_substraction(x, tmp) ^ x);
        //cout<<x<<" "<<bits_substraction(x, tmp)<<" "<<rows[i]<<endl;
    }
}

void fixup(string B, int start_i, int end_i, vector<bitset<M> > rows, vector<bitset<M> > alphabet_strings) {
    bitset<M> correct;
    bitset<M> x;
    bitset<M> tmp;
    for(int i = start_i; i < end_i; i++) {  // i start from 1 so B[i-1]
        x = rows[i-1] | get_alphabet_string(B[i-1], alphabet_strings);
        tmp = rows[i-1] << 1;
        tmp[0] = 1;
        correct = x & (bits_substraction(x, tmp) ^ x);

        bool stop = is_parallel(correct, rows[i]);
        rows[i] = correct;
        if(stop) {
            cout<<"stop at "<<i<<endl;
            return;
        }
    }
}


void exam_alphabet_strings(vector<bitset<M> > alphabet_strings) {
    for(int i = 0; i < alphabet_strings.size(); i++) {
        cout<<alphabet[i]<< ": "<<alphabet_strings[i]<<endl;
    }
}

string random_string(int length) {
    srand(time(0));
    string s(length, 'A');
    const char alphabet[] = {'A', 'C', 'G', 'T'};
    for(int i = 0; i < length; i++) {
        s[i] = alphabet[(rand() % 4)];
    }
    return s;
}

int main(int argc, char *argv[]) {
    //string A = "GTCTTACATCCGTTCG";
    //string B = "TAGCTTAAGATCTTGT";
    string A = random_string(M);
    string B = random_string(N);
    //printf("A:%s\nB:%s\n", A.c_str(), B.c_str());


    vector<bitset<M> > alphabet_strings = compute_alphabet_strings(A);
    /*
    vector<bitset<M> > rows(N+1);
    exam_alphabet_strings(alphabet_strings);
    compute_matrix(B, 1, N, rows, alphabet_strings);
    compute_matrix(B, N/4, N+1, rows, alphabet_strings);
    fixup(B, N/4, N+1, rows, alphabet_strings);
    */

    clock_t t = clock();
    compute_matrix(B, 1, N, alphabet_strings);
    double seconds = (clock() - t) / (double) CLOCKS_PER_SEC;
    double gcpus = M * 1.0 / G * N / seconds;
    printf("time:%f, GCPUS: %f\n", seconds, gcpus);
    return 0;
}
