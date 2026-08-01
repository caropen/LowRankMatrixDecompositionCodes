#!/bin/bash
source /opt/intel/oneapi/setvars.sh
module use /opt/nvidia/hpc_sdk/modulefiles/
module load nvhpc
export MKL_THREADING_LAYER=GNU

nvcc driver_mkl_and_cublas1.c rank_revealing_algorithms_mkl_and_cublas.c matrix_vector_functions_mkl_and_cublas.c -o driver_mkl_and_cublas1 -Xcompiler -fopenmp -lmkl_rt -lcublas -lcudart
