#ifndef MATRIX_VECTOR_FUNCTIONS_MKL_AND_CUBLAS_SINGLE_H
#define MATRIX_VECTOR_FUNCTIONS_MKL_AND_CUBLAS_SINGLE_H

#include <cublas_v2.h>

extern cublasHandle_t handle_single;

typedef struct {
    int nrows;
    int ncols;
    float *d;
    float *ddev;
} smat;

typedef struct {
    int nrows;
    float *d;
} svec;

smat *smatrix_new(int nrows, int ncols);
svec *svector_new(int nrows);
void smatrix_delete(smat *matrix);
void svector_delete(svec *vector);

void smatrix_copy(smat *destination, const smat *source);
void smatrix_build_transpose(smat *transpose, const smat *matrix);
void smatrix_sub(smat *left, const smat *right);

void sinitialize_random_matrix(smat *matrix);
void smatrix_matrix_mult(smat *left, smat *right, smat *product);
void smatrix_transpose_matrix_mult(smat *left, smat *right, smat *product);
void smatrix_matrix_transpose_mult(smat *left, smat *right, smat *product);

void smatrix_get_selected_columns(const smat *matrix, const int *indices, smat *columns);
void smatrix_set_selected_columns(smat *matrix, const int *indices, const smat *columns);
void smatrix_set_selected_rows(smat *matrix, const int *indices, const smat *rows);
void sresize_matrix_by_columns(smat **matrix, int ncols);
void sresize_matrix_by_rows(smat **matrix, int nrows);

float sget_matrix_frobenius_norm(const smat *matrix);
float sget_percent_error_between_two_mats(const smat *left, const smat *right);
void sQR_factorization_getQ(const smat *matrix, smat *q_factor);

#endif
