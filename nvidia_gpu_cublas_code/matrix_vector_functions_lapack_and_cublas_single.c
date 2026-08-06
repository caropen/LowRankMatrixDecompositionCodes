#include "matrix_vector_functions_lapack_and_cublas_single.h"

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

cublasHandle_t handle_single;
static lapack_int random_seed_single[4] = { 1, 2, 3, 5 };

static size_t smatrix_element_count(const smat *matrix)
{
    return (size_t)matrix->nrows * (size_t)matrix->ncols;
}

static void smatrix_elems_copy_to_device(smat *matrix)
{
    cudaMemcpy(matrix->ddev, matrix->d,
               smatrix_element_count(matrix) * sizeof(float),
               cudaMemcpyHostToDevice);
}

static void smatrix_elems_copy_from_device(smat *matrix)
{
    cudaMemcpy(matrix->d, matrix->ddev,
               smatrix_element_count(matrix) * sizeof(float),
               cudaMemcpyDeviceToHost);
}

smat *smatrix_new(int nrows, int ncols)
{
    size_t count = (size_t)nrows * (size_t)ncols;
    smat *matrix = (smat *)malloc(sizeof(*matrix));

    matrix->nrows = nrows;
    matrix->ncols = ncols;
    matrix->d = (float *)calloc(count, sizeof(float));
    cudaMalloc((void **)&matrix->ddev, count * sizeof(float));
    cudaMemset(matrix->ddev, 0, count * sizeof(float));
    return matrix;
}

svec *svector_new(int nrows)
{
    svec *vector = (svec *)malloc(sizeof(*vector));

    vector->nrows = nrows;
    vector->d = (float *)calloc((size_t)nrows, sizeof(float));
    return vector;
}

void smatrix_delete(smat *matrix)
{
    if (matrix == NULL) {
        return;
    }
    free(matrix->d);
    cudaFree(matrix->ddev);
    free(matrix);
}

void svector_delete(svec *vector)
{
    if (vector == NULL) {
        return;
    }
    free(vector->d);
    free(vector);
}

void smatrix_copy(smat *destination, const smat *source)
{
    memcpy(destination->d, source->d,
           smatrix_element_count(source) * sizeof(float));
}

void smatrix_build_transpose(smat *transpose, const smat *matrix)
{
    int i, j;

    #pragma omp parallel for private(i)
    for (j = 0; j < matrix->ncols; ++j) {
        for (i = 0; i < matrix->nrows; ++i) {
            transpose->d[(size_t)i * transpose->nrows + j] =
                matrix->d[(size_t)j * matrix->nrows + i];
        }
    }
}

void smatrix_sub(smat *left, const smat *right)
{
    size_t i;
    size_t count = smatrix_element_count(left);

    #pragma omp parallel for
    for (i = 0; i < count; ++i) {
        left->d[i] -= right->d[i];
    }
}

void sinitialize_random_matrix(smat *matrix)
{
    size_t offset = 0;
    size_t count = smatrix_element_count(matrix);

    while (offset < count) {
        size_t remaining = count - offset;
        lapack_int chunk = remaining > (size_t)INT_MAX
                         ? (lapack_int)INT_MAX
                         : (lapack_int)remaining;
        LAPACKE_slarnv(3, random_seed_single, chunk, matrix->d + offset);
        offset += (size_t)chunk;
    }
}

void smatrix_matrix_mult(smat *left, smat *right, smat *product)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    smatrix_elems_copy_to_device(left);
    smatrix_elems_copy_to_device(right);
    cublasSgemm(handle_single, CUBLAS_OP_N, CUBLAS_OP_N,
                left->nrows, right->ncols, left->ncols,
                &alpha, left->ddev, left->nrows,
                right->ddev, right->nrows,
                &beta, product->ddev, product->nrows);
    smatrix_elems_copy_from_device(product);
}

void smatrix_transpose_matrix_mult(smat *left, smat *right, smat *product)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    smatrix_elems_copy_to_device(left);
    smatrix_elems_copy_to_device(right);
    cublasSgemm(handle_single, CUBLAS_OP_T, CUBLAS_OP_N,
                left->ncols, right->ncols, left->nrows,
                &alpha, left->ddev, left->nrows,
                right->ddev, right->nrows,
                &beta, product->ddev, product->nrows);
    smatrix_elems_copy_from_device(product);
}

void smatrix_matrix_transpose_mult(smat *left, smat *right, smat *product)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    smatrix_elems_copy_to_device(left);
    smatrix_elems_copy_to_device(right);
    cublasSgemm(handle_single, CUBLAS_OP_N, CUBLAS_OP_T,
                left->nrows, right->nrows, left->ncols,
                &alpha, left->ddev, left->nrows,
                right->ddev, right->nrows,
                &beta, product->ddev, product->nrows);
    smatrix_elems_copy_from_device(product);
}

void smatrix_get_selected_columns(const smat *matrix, const int *indices, smat *columns)
{
    int j;

    #pragma omp parallel for
    for (j = 0; j < columns->ncols; ++j) {
        memcpy(columns->d + (size_t)j * columns->nrows,
               matrix->d + (size_t)indices[j] * matrix->nrows,
               (size_t)matrix->nrows * sizeof(float));
    }
}

void smatrix_set_selected_columns(smat *matrix, const int *indices, const smat *columns)
{
    int j;

    #pragma omp parallel for
    for (j = 0; j < columns->ncols; ++j) {
        memcpy(matrix->d + (size_t)indices[j] * matrix->nrows,
               columns->d + (size_t)j * columns->nrows,
               (size_t)matrix->nrows * sizeof(float));
    }
}

void smatrix_set_selected_rows(smat *matrix, const int *indices, const smat *rows)
{
    int i, j;

    #pragma omp parallel for private(i)
    for (j = 0; j < rows->ncols; ++j) {
        for (i = 0; i < rows->nrows; ++i) {
            matrix->d[(size_t)j * matrix->nrows + indices[i]] =
                rows->d[(size_t)j * rows->nrows + i];
        }
    }
}

void sresize_matrix_by_columns(smat **matrix, int ncols)
{
    smat *resized = smatrix_new((*matrix)->nrows, ncols);

    memcpy(resized->d, (*matrix)->d,
           (size_t)(*matrix)->nrows * ncols * sizeof(float));
    smatrix_delete(*matrix);
    *matrix = resized;
}

void sresize_matrix_by_rows(smat **matrix, int nrows)
{
    int j;
    smat *resized = smatrix_new(nrows, (*matrix)->ncols);

    #pragma omp parallel for
    for (j = 0; j < (*matrix)->ncols; ++j) {
        memcpy(resized->d + (size_t)j * nrows,
               (*matrix)->d + (size_t)j * (*matrix)->nrows,
               (size_t)nrows * sizeof(float));
    }
    smatrix_delete(*matrix);
    *matrix = resized;
}

float sget_matrix_frobenius_norm(const smat *matrix)
{
    size_t i;
    size_t count = smatrix_element_count(matrix);
    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < count; ++i) {
        sum += matrix->d[i] * matrix->d[i];
    }
    return sqrtf(sum);
}

float sget_percent_error_between_two_mats(const smat *left, const smat *right)
{
    float left_norm;
    float error_norm;
    smat *difference = smatrix_new(left->nrows, left->ncols);

    smatrix_copy(difference, left);
    smatrix_sub(difference, right);
    left_norm = sget_matrix_frobenius_norm(left);
    error_norm = sget_matrix_frobenius_norm(difference);
    smatrix_delete(difference);
    return 100.0f * error_norm / left_norm;
}

void sQR_factorization_getQ(const smat *matrix, smat *q_factor)
{
    int k = matrix->nrows < matrix->ncols ? matrix->nrows : matrix->ncols;
    svec *tau = svector_new(k);

    smatrix_copy(q_factor, matrix);
    LAPACKE_sgeqrf(LAPACK_COL_MAJOR, matrix->nrows, matrix->ncols,
                   q_factor->d, matrix->nrows, tau->d);
    LAPACKE_sorgqr(LAPACK_COL_MAJOR, matrix->nrows, matrix->ncols,
                   k, q_factor->d, matrix->nrows, tau->d);
    svector_delete(tau);
}
