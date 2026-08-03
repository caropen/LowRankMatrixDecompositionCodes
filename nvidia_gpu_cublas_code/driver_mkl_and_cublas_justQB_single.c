/*
   Test single-precision randQB_pb_new on an SVD-defined matrix using cuBLAS
   and Intel MKL.
*/

#include "rank_revealing_algorithms_mkl_and_cublas_single.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int min_int(int left, int right)
{
    return left < right ? left : right;
}

static void print_usage(const char *program_name)
{
    fprintf(stderr, "Usage: %s <rows> <cols> [--tolerance <value>] [--kstep <value>]\n", program_name);
    fprintf(stderr, "  rows, cols: positive integers greater than 1\n");
    fprintf(stderr, "  --tolerance: optional positive absolute tolerance\n");
    fprintf(stderr, "  --kstep: optional positive QB block size that divides min(rows, cols)\n");
}

static int parse_dimension(const char *text, int *value)
{
    char *end = NULL;
    intmax_t parsed;

    errno = 0;
    parsed = strtoimax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed <= 0 || parsed > INT_MAX) {
        return 0;
    }
    *value = (int)parsed;
    return 1;
}

static int parse_tolerance(const char *text, float *value)
{
    char *end = NULL;
    float parsed;

    errno = 0;
    parsed = strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !isfinite(parsed) || parsed <= 0.0f) {
        return 0;
    }
    *value = parsed;
    return 1;
}

static smat *generate_svd_defined_matrix(int m, int n)
{
    int i, j;
    int min_mn = min_int(m, n);
    smat *u_random, *v_random, *u, *v, *matrix;
    svec *sigma;

    u_random = smatrix_new(m, min_mn);
    u = smatrix_new(m, min_mn);
    sinitialize_random_matrix(u_random);
    sQR_factorization_getQ(u_random, u);
    smatrix_delete(u_random);

    v_random = smatrix_new(n, min_mn);
    v = smatrix_new(n, min_mn);
    sinitialize_random_matrix(v_random);
    sQR_factorization_getQ(v_random, v);
    smatrix_delete(v_random);

    sigma = svector_new(min_mn);
    for (j = 0; j < min_mn; ++j) {
        sigma->d[j] = (float)min_mn / powf((float)(j + 1), 0.85f);
    }

    #pragma omp parallel for private(i)
    for (j = 0; j < min_mn; ++j) {
        float sigma_j = sigma->d[j];
        for (i = 0; i < m; ++i) {
            u->d[(size_t)j * m + i] *= sigma_j;
        }
    }

    printf("largest prescribed singular value = %g\n", sigma->d[0]);
    printf("smallest prescribed singular value = %g\n", sigma->d[min_mn - 1]);

    matrix = smatrix_new(m, n);
    smatrix_matrix_transpose_mult(u, v, matrix);

    svector_delete(sigma);
    smatrix_delete(u);
    smatrix_delete(v);
    return matrix;
}

int main(int argc, char **argv)
{
    int m, n, min_mn, kstep, nstep, q, s;
    int frank, argi, tolerance_supplied, kstep_supplied;
    uintmax_t nentries;
    float tolerance = 0.0f;
    double start_time, end_time, gpu_time;
    smat *matrix, *q_factor, *b_factor;
    cublasStatus_t cublas_status;

    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!parse_dimension(argv[1], &m) || !parse_dimension(argv[2], &n)) {
        fprintf(stderr, "Error: rows and cols must be positive integers within the supported range.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    min_mn = min_int(m, n);
    if (min_mn < 2) {
        fprintf(stderr, "Error: rows and cols must both be greater than 1.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if ((uintmax_t)m > SIZE_MAX / sizeof(float) / (uintmax_t)n) {
        fprintf(stderr, "Error: the requested matrix dimensions overflow the addressable allocation size.\n");
        return EXIT_FAILURE;
    }

    kstep = 200;
    if (kstep > min_mn / 2) {
        kstep = min_mn / 10;
        if (kstep < 1) {
            kstep = 1;
        }
    }

    tolerance_supplied = 0;
    kstep_supplied = 0;
    argi = 3;
    while (argi < argc) {
        if (strcmp(argv[argi], "--tolerance") == 0) {
            if (tolerance_supplied) {
                fprintf(stderr, "Error: --tolerance may only be specified once.\n");
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            if (argi + 1 >= argc || strncmp(argv[argi + 1], "--", 2) == 0) {
                fprintf(stderr, "Error: --tolerance requires a value.\n");
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            if (!parse_tolerance(argv[argi + 1], &tolerance)) {
                fprintf(stderr, "Error: tolerance must be a positive finite number.\n");
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            tolerance_supplied = 1;
            argi += 2;
        }
        else if (strcmp(argv[argi], "--kstep") == 0) {
            if (kstep_supplied) {
                fprintf(stderr, "Error: --kstep may only be specified once.\n");
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            if (argi + 1 >= argc || strncmp(argv[argi + 1], "--", 2) == 0) {
                fprintf(stderr, "Error: --kstep requires a value.\n");
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            if (!parse_dimension(argv[argi + 1], &kstep)) {
                fprintf(stderr, "Error: kstep must be a positive integer within the supported range.\n");
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            kstep_supplied = 1;
            argi += 2;
        }
        else {
            fprintf(stderr, "Error: unknown argument '%s'.\n", argv[argi]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (kstep_supplied && kstep > min_mn / 2) {
        fprintf(stderr, "Error: kstep must not exceed min(rows, cols)/2.\n");
        return EXIT_FAILURE;
    }
    if (kstep_supplied && min_mn % kstep != 0) {
        fprintf(stderr, "Error: kstep must divide min(rows, cols) exactly.\n");
        return EXIT_FAILURE;
    }
    if (!tolerance_supplied && min_mn % kstep != 0) {
        fprintf(stderr, "Error: the default kstep does not divide min(rows, cols); provide a dividing --kstep for full-rank mode.\n");
        return EXIT_FAILURE;
    }

    nentries = (uintmax_t)m * (uintmax_t)n;
    printf("matrix precision = single\n");
    printf("matrix size = %d by %d\n", m, n);
    printf("number of entries = %" PRIuMAX "\n", nentries);
    printf("QB block size = %d\n", kstep);
    if (tolerance_supplied) {
        printf("QB stopping mode = absolute tolerance\n");
        printf("absolute stopping tolerance = %g\n", tolerance);
    }
    else {
        printf("QB stopping mode = full rank\n");
    }

    printf("Initializing cuBLAS\n");
    cublas_status = cublasCreate(&handle_single);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error: cublasCreate failed with status %d.\n", (int)cublas_status);
        return EXIT_FAILURE;
    }

    printf("generating single-precision SVD-defined matrix..\n");
    start_time = omp_get_wtime();
    matrix = generate_svd_defined_matrix(m, n);
    end_time = omp_get_wtime();
    printf("matrix_generation_seconds %11.6f\n", end_time - start_time);

    q = 1;
    s = 2;
    if (tolerance_supplied) {
        nstep = -1;
        printf("call randQB_pb_new_single in TOL mode with block size %d..\n", kstep);
    }
    else {
        nstep = min_mn / kstep;
        printf("call randQB_pb_new_single to full rank with block size %d..\n", kstep);
    }

    start_time = omp_get_wtime();
    randQB_pb_new_single(matrix, kstep, nstep, tolerance, q, s,
                         &frank, &q_factor, &b_factor);
    end_time = omp_get_wtime();
    gpu_time = end_time - start_time;
    printf("qb_rangefinder_seconds %11.6f\n", gpu_time);
    printf("output frank = %d\n", frank);
    printf("norm(Q) = %f, norm(B) = %f\n",
           sget_matrix_frobenius_norm(q_factor),
           sget_matrix_frobenius_norm(b_factor));
    suse_QB_decomp_for_approximation(matrix, q_factor, b_factor);

    printf("delete and exit..\n");
    smatrix_delete(matrix);
    smatrix_delete(q_factor);
    smatrix_delete(b_factor);

    printf("Shutting down cuBLAS\n");
    cublas_status = cublasDestroy(handle_single);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error: cublasDestroy failed with status %d.\n", (int)cublas_status);
        return EXIT_FAILURE;
    }

    printf("RANDQB_RESULT,%lld,%lld,%lld,%.9e,%d,%lld,%.6f,%s\n",
           (long long)m, (long long)n, (long long)kstep,
           (double)tolerance, nstep, (long long)frank, gpu_time, "ok");

    return EXIT_SUCCESS;
}
