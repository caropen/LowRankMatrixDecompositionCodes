NVHPC_ROOT      ?= /opt/nvidia/hpc_sdk/Linux_x86_64/25.11
CUDADIR         ?= $(NVHPC_ROOT)/cuda
CUDAMATHDIR     ?= $(NVHPC_ROOT)/math_libs
NVPLDIR         ?= /opt/nvidia/nvpl
OPENBLAS_INCDIR ?= /usr/include/openblas
OPENBLAS_LIBDIR ?= /usr/lib64
USE_MKL         ?= 0
USE_NVPL        ?= 0

CUDA_CXX ?= nvcc

ifeq ($(USE_MKL)$(USE_NVPL),11)
$(error USE_MKL and USE_NVPL cannot both be 1)
endif

# CUDA, cuBLAS, GCC OpenMP, and OpenBLAS are the default.
ifeq ($(USE_MKL),1)
ARCH_FLAGS  =
OPENMP_FLAG = -Xcompiler -fopenmp
BLAS_CFLAGS = -DUSE_MKL -I$(MKLROOT)/include
BLAS_LIBS   = -L$(MKLROOT)/lib -lmkl_rt
else ifeq ($(USE_NVPL),1)
ARCH_FLAGS  = -Xcompiler -mcpu=grace
OPENMP_FLAG = -Xcompiler -fopenmp
BLAS_CFLAGS = -DUSE_NVPL -I$(NVPLDIR)/include
BLAS_LIBS   = -L$(NVPLDIR)/lib \
	      -lnvpl_lapack_lp64_gomp -lnvpl_blas_lp64_gomp -lgomp
else
ARCH_FLAGS  =
OPENMP_FLAG = -Xcompiler -fopenmp
BLAS_CFLAGS = -I$(OPENBLAS_INCDIR)
BLAS_LIBS   = -L$(OPENBLAS_LIBDIR) -lopenblas
endif

CUDA_CFLAGS = -I$(CUDADIR)/include
CUDA_LIBS   = -L$(CUDADIR)/lib64 -L$(CUDAMATHDIR)/lib64 \
	      -lcublas -lcudart
CPPFLAGS    = $(CUDA_CFLAGS) $(BLAS_CFLAGS)
NVCCFLAGS   = -O3 $(ARCH_FLAGS) $(OPENMP_FLAG)
LDLIBS      = $(BLAS_LIBS) $(CUDA_LIBS) -lm

TARGET = driver_lapack_and_cublas_justQB_single
SOURCES = driver_lapack_and_cublas_justQB_single.c \
	  matrix_vector_functions_lapack_and_cublas_single.c \
	  rank_revealing_algorithms_lapack_and_cublas_single.c
HEADERS = matrix_vector_functions_lapack_and_cublas_single.h \
	  rank_revealing_algorithms_lapack_and_cublas_single.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CUDA_CXX) $(CPPFLAGS) $(NVCCFLAGS) $(SOURCES) $(LDLIBS) -o $@

clean:
	$(RM) $(TARGET)
