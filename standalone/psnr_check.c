#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char **argv) {
    if(argc < 4) { fprintf(stderr, "Usage: psnr_check output.bin clean.bin npix\n"); return 1; }
    size_t npix = atol(argv[3]);
    float *out = malloc(npix * sizeof(float));
    float *cln = malloc(npix * sizeof(float));
    FILE *f1 = fopen(argv[1], "rb"); fread(out, sizeof(float), npix, f1); fclose(f1);
    FILE *f2 = fopen(argv[2], "rb"); fread(cln, sizeof(float), npix, f2); fclose(f2);
    double mse = 0.0;
    for(size_t i = 0; i < npix; i++) {
        double d = (double)out[i] - (double)cln[i];
        mse += d * d;
    }
    mse /= (double)npix;
    double psnr = 10.0 * log10(1.0 / mse);
    printf("MSE=%.8f PSNR=%.4f dB (npix=%zu)\n", mse, psnr, npix);
    free(out); free(cln);
    return 0;
}
