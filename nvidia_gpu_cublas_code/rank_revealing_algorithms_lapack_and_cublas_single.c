#include "rank_revealing_algorithms_lapack_and_cublas_single.h"

#include <stdio.h>
#include <stdlib.h>

static int min_int(int left, int right)
{
    return left < right ? left : right;
}

void randQB_pb_new_single(smat *M, int kstep, int nstep, float tolerance,
                          int q, int s, int *frank, smat **Q, smat **B)
{
    int i, j, step;
    int m = M->nrows;
    int n = M->ncols;
    int tolerance_mode;
    int l;
    int *local_indices;
    smat *A, *random_matrix, *random_block, *sample, *q_block, *b_block;
    smat *q_block_b_block, *at_q_block, *at_q_block_orthogonal;
    smat *q_block_transpose_a;

    if (kstep > min_int(m, n) / 2) {
        kstep = min_int(m, n) / 10;
        if (kstep < 1) {
            kstep = 1;
        }
        printf("kstep resized to %d\n", kstep);
    }

    tolerance_mode = nstep <= 0;
    if (tolerance_mode) {
        nstep = min_int(m, n) / kstep;
        printf("using TOL mode\n");
    }
    else {
        printf("using rank mode\n");
    }

    l = kstep * nstep;
    random_matrix = smatrix_new(n, l);
    sinitialize_random_matrix(random_matrix);

    A = smatrix_new(m, n);
    *Q = smatrix_new(m, l);
    *B = smatrix_new(l, n);
    random_block = smatrix_new(n, kstep);
    sample = smatrix_new(m, kstep);
    q_block = smatrix_new(m, kstep);
    b_block = smatrix_new(kstep, n);
    q_block_b_block = smatrix_new(m, n);
    at_q_block = smatrix_new(n, kstep);
    at_q_block_orthogonal = smatrix_new(n, kstep);
    q_block_transpose_a = smatrix_new(kstep, n);
    smatrix_copy(A, M);

    for (step = 0; step < nstep; ++step) {
        local_indices = (int *)malloc((size_t)kstep * sizeof(int));
        for (i = 0; i < kstep; ++i) {
            local_indices[i] = kstep * step + i;
        }

        smatrix_get_selected_columns(random_matrix, local_indices, random_block);
        smatrix_matrix_mult(A, random_block, sample);

        for (j = 1; j <= q; ++j) {
            if ((2 * j - 2) % s == 0) {
                sQR_factorization_getQ(sample, q_block);
                smatrix_transpose_matrix_mult(q_block, A, q_block_transpose_a);
                smatrix_build_transpose(at_q_block, q_block_transpose_a);
            }
            else {
                smatrix_transpose_matrix_mult(A, sample, at_q_block);
            }

            if ((2 * j - 1) % s == 0) {
                sQR_factorization_getQ(at_q_block, at_q_block_orthogonal);
                smatrix_matrix_mult(A, at_q_block_orthogonal, sample);
            }
            else {
                smatrix_matrix_mult(A, at_q_block, sample);
            }
        }

        sQR_factorization_getQ(sample, q_block);

        if (step > 0 && step % 2 == 0) {
            int previous_rank = step * kstep;
            int *global_indices = (int *)malloc((size_t)previous_rank * sizeof(int));
            smat *previous_q = smatrix_new(m, previous_rank);
            smat *projection_coefficients = smatrix_new(previous_rank, kstep);
            smat *projection = smatrix_new(m, kstep);

            for (i = 0; i < previous_rank; ++i) {
                global_indices[i] = i;
            }
            smatrix_get_selected_columns(*Q, global_indices, previous_q);
            smatrix_transpose_matrix_mult(previous_q, q_block, projection_coefficients);
            smatrix_matrix_mult(previous_q, projection_coefficients, projection);
            smatrix_copy(sample, q_block);
            smatrix_sub(sample, projection);
            sQR_factorization_getQ(sample, q_block);

            smatrix_delete(projection);
            smatrix_delete(projection_coefficients);
            smatrix_delete(previous_q);
            free(global_indices);
        }

        smatrix_transpose_matrix_mult(q_block, A, b_block);
        smatrix_matrix_mult(q_block, b_block, q_block_b_block);
        smatrix_sub(A, q_block_b_block);
        smatrix_set_selected_columns(*Q, local_indices, q_block);
        smatrix_set_selected_rows(*B, local_indices, b_block);
        free(local_indices);

        *frank = (step + 1) * kstep;
        if (tolerance_mode) {
            float residual_norm = sget_matrix_frobenius_norm(A);
            printf("at step %d, norm(A^{%d}) = %f\n", step, step, residual_norm);
            if (residual_norm < tolerance) {
                break;
            }
        }
    }

    if (tolerance_mode) {
        sresize_matrix_by_columns(Q, *frank);
        sresize_matrix_by_rows(B, *frank);
    }

    smatrix_delete(A);
    smatrix_delete(random_matrix);
    smatrix_delete(random_block);
    smatrix_delete(sample);
    smatrix_delete(q_block);
    smatrix_delete(b_block);
    smatrix_delete(q_block_b_block);
    smatrix_delete(at_q_block);
    smatrix_delete(at_q_block_orthogonal);
    smatrix_delete(q_block_transpose_a);
}

void suse_QB_decomp_for_approximation(smat *M, smat *Q, smat *B)
{
    smat *approximation = smatrix_new(M->nrows, M->ncols);

    smatrix_matrix_mult(Q, B, approximation);
    printf("norm(M,fro) = %f\n", sget_matrix_frobenius_norm(M));
    printf("norm(P,fro) = %f\n", sget_matrix_frobenius_norm(approximation));
    printf("percent error = %f\n",
           sget_percent_error_between_two_mats(M, approximation));
    smatrix_delete(approximation);
}
