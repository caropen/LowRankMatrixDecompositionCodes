/*
   Test randomized QB decomposition in tolerance mode on an SVD-defined matrix.
*/

#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#elif defined(USE_NVPL)
#include <nvpl_blas_cblas.h>
#include <nvpl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "rank_revealing_algorithms_lapack.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static void print_usage(const char *program_name)
{
    fprintf(stderr, "Usage: %s <rows> <cols> [--tolerance <value>] [--kstep <value>]\n", program_name);
    fprintf(stderr, "  rows, cols: positive integers greater than 1\n");
    fprintf(stderr, "  --tolerance: optional positive absolute tolerance\n");
    fprintf(stderr, "  --kstep: optional positive QB block size that divides min(rows, cols)\n");
}


static int parse_dimension(const char *text, myint64 *value)
{
    char *end = NULL;
    intmax_t parsed;

    errno = 0;
    parsed = strtoimax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        return 0;
    }

    *value = (myint64)parsed;
    return 1;
}


static int parse_tolerance(const char *text, float *value)
{
    char *end = NULL;
    float parsed;

    errno = 0;
    parsed = strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !isfinite(parsed) || parsed <= 0.0f) {
        return 0;
    }

    *value = parsed;
    return 1;
}


static mat *generate_svd_defined_matrix(myint64 m, myint64 n)
{
    myint64 i, j;
    myint64 min_mn = min(m, n);
    mat *U_random, *V_random, *U, *V, *A;
    vec *sigma;

    U_random = matrix_new(m, min_mn);
    U = matrix_new(m, min_mn);
    initialize_random_matrix(U_random);
    QR_factorization_getQ(U_random, U);
    matrix_delete(U_random);

    V_random = matrix_new(n, min_mn);
    V = matrix_new(n, min_mn);
    initialize_random_matrix(V_random);
    QR_factorization_getQ(V_random, V);
    matrix_delete(V_random);

    sigma = vector_new(min_mn);
    for (j = 0; j < min_mn; ++j) {
        sigma->d[j] = (float)min_mn / powf((float)(j + 1), 0.85f);
    }

    #pragma omp parallel for private(i)
    for (j = 0; j < min_mn; ++j) {
        float sigma_j = sigma->d[j];
        for (i = 0; i < m; ++i) {
            U->d[j*m + i] *= sigma_j;
        }
    }

    printf("largest prescribed singular value = %g\n", sigma->d[0]);
    printf("smallest prescribed singular value = %g\n", sigma->d[min_mn - 1]);

    A = matrix_new(m, n);
    matrix_matrix_transpose_mult(U, V, A);

    vector_delete(sigma);
    matrix_delete(U);
    matrix_delete(V);

    return A;
}


int main(int argc, char **argv)
{
    myint64 m, n, min_mn, kstep, nstep, q, s;
    myint64 frank, numnnz;
    float TOL;
    mat *A, *Q, *B;
    double start_time, end_time, cpu_time;
    int argi, tolerance_supplied, kstep_supplied;

    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!parse_dimension(argv[1], &m) || !parse_dimension(argv[2], &n)) {
        fprintf(stderr, "Error: rows and cols must be positive integers within the LP64 LAPACK range.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    min_mn = min(m, n);
    if (min_mn < 2) {
        fprintf(stderr, "Error: rows and cols must both be greater than 1.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if ((uintmax_t)m > SIZE_MAX / sizeof(float) / (uintmax_t)n) {
        fprintf(stderr, "Error: the requested matrix dimensions overflow the addressable allocation size.\n");
        return EXIT_FAILURE;
    }

    TOL = 0.0f;
    kstep = 200;
    if (kstep > min_mn/2) {
        kstep = min_mn/10;
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
            if (!parse_tolerance(argv[argi + 1], &TOL)) {
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
                fprintf(stderr, "Error: kstep must be a positive integer within the LP64 LAPACK range.\n");
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

    if (kstep_supplied && kstep > min_mn/2) {
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

    numnnz = m*n;
    printf("matrix size = %" PRId64 " by %" PRId64 "\n", m, n);
    printf("number of entries = %" PRId64 "\n", numnnz);
    printf("QB block size = %" PRId64 "\n", kstep);
    if (tolerance_supplied) {
        printf("QB stopping mode = absolute tolerance\n");
        printf("absolute stopping tolerance = %g\n", TOL);
    }
    else {
        printf("QB stopping mode = full rank\n");
    }

    printf("generating SVD-defined matrix..\n");
    start_time = omp_get_wtime();
    A = generate_svd_defined_matrix(m, n);
    end_time = omp_get_wtime();
    printf("matrix_generation_seconds %11.6f\n", end_time - start_time);

    q = 1; // power scheme power
    s = 2; // power scheme orthogonalization amount

    if (tolerance_supplied) {
        nstep = -1;
        printf("call QB decomp in TOL mode with block size %" PRId64 "..\n", kstep);
    }
    else {
        nstep = min_mn/kstep;
        printf("call QB decomp to full rank with block size %" PRId64 "..\n", kstep);
    }
    start_time = omp_get_wtime();
    randQB_pb2(A, kstep, nstep, TOL, q, s, &frank, &Q, &B);
    end_time = omp_get_wtime();
    cpu_time = end_time - start_time;
    printf("qb_rangefinder_seconds %11.6f\n", cpu_time);
    printf("output frank = %" PRId64 "\n", frank);
    printf("norm(Q) = %f, norm(B) = %f\n", get_matrix_frobenius_norm(Q), get_matrix_frobenius_norm(B));
    use_QB_decomp_for_approximation(A, Q, B);

    printf("delete and exit..\n");
    matrix_delete(A);
    matrix_delete(Q);
    matrix_delete(B);

    printf("RANDQB_RESULT,%lld,%lld,%lld,%.9e,%d,%lld,%.6f,%s\n",
           (long long)m, (long long)n, (long long)kstep,
           (double)TOL, (int)nstep, (long long)frank, cpu_time, "ok");

    return EXIT_SUCCESS;
}
