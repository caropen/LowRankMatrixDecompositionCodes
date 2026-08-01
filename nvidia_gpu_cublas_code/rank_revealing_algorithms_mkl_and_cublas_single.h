#ifndef RANK_REVEALING_ALGORITHMS_MKL_AND_CUBLAS_SINGLE_H
#define RANK_REVEALING_ALGORITHMS_MKL_AND_CUBLAS_SINGLE_H

#include "matrix_vector_functions_mkl_and_cublas_single.h"

void randQB_pb_new_single(smat *M, int kstep, int nstep, float tolerance,
                          int q, int s, int *frank, smat **Q, smat **B);
void suse_QB_decomp_for_approximation(smat *M, smat *Q, smat *B);

#endif
