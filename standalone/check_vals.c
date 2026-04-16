#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    float buf[20];
    fread(buf, sizeof(float), 20, f); fclose(f);
    for(int i = 0; i < 20; i++) printf("[%d] %.6f\n", i, buf[i]);
    return 0;
}
