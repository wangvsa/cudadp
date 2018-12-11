#include <iostream>
#include <string>
#include <stdio.h>
using namespace std;


string read_fasta_file(char *path) {
    string sequence;
    FILE* fp;
    char buf[100];
    if ((fp = fopen(path, "r")) == NULL) {
        perror("fopen source-file");
    }

    while(fgets(buf, sizeof(buf), fp) != NULL) {
        buf[strlen(buf) - 1] = '\0'; // eat the newline fgets() stores
        if(buf[0] == '>') {           // skip the header
            printf("%s\n", buf);
            continue;
        }
        sequence += buf;
    }
    fclose(fp);

    cout<<"sequence length: "<<sequence.length()<<endl;
    return sequence;
}
