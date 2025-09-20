//
// Created by Sreeram Raghammudi on 9/20/25.
//


//
// Created by Sreeram Raghammudi on 9/20/25.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0f;
    for (long j = 0; j < N; j += 4) {
        R += pA[j] * pB[j]
           + pA[j+1] * pB[j+1]
           + pA[j+2] * pB[j+2]
           + pA[j+3] * pB[j+3];
    }
    return R;
}


double elapsed_seconds(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <vector_size> <repetitions>\n", argv[0]);
        return EXIT_FAILURE;
    }

    long N = atol(argv[1]);
    int repetitions = atoi(argv[2]);

    // Allocate vectors
    float *A = (float *) malloc(N * sizeof(float));
    float *B = (float *) malloc(N * sizeof(float));
    if (!A || !B) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    for (long i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    double *times = (double *) malloc(repetitions * sizeof(double));
    volatile float result = 0.0f;

    struct timespec start, end;

    for (int r = 0; r < repetitions; r++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        result = dpunroll(N, A, B);
        clock_gettime(CLOCK_MONOTONIC, &end);
        times[r] = elapsed_seconds(start, end);
    }

    // Compute mean time using the second half of repetitions
    double sum = 0.0;
    int half = repetitions / 2;
    for (int r = half; r < repetitions; r++) {
        sum += times[r];
    }
    double mean_time = sum / (repetitions - half);

    // Compute bandwidth (GB/s)
    double bytes = 2.0 * N * sizeof(float); // 2 arrays accessed
    double bandwidth = (bytes / mean_time) / 1e9;

    // Compute throughput (FLOP/s)
    double flops = (2.0 * N) / mean_time; // ~2N FLOPs

    // Print result with nanosecond precision
    printf("N: %ld <T>: %.9f sec B: %.3f GB/sec F: %.3f FLOP/sec\n",
           N, mean_time, bandwidth, flops);

    free(A);
    free(B);
    free(times);
    return EXIT_SUCCESS;
}
