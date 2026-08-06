NVPLDIR         ?= /opt/nvidia/nvpl
OPENBLAS_INCDIR ?= /usr/include/openblas
OPENBLAS_LIBDIR ?= /usr/lib64
USE_MKL         ?= 0
USE_NVPL        ?= 0

ifeq ($(USE_MKL)$(USE_NVPL),11)
$(error USE_MKL and USE_NVPL cannot both be 1)
endif

# GCC and OpenBLAS are the default.
ifeq ($(USE_MKL),1)
CC          = icx
CXX         = icpx
FORT        = ifx
ARCH_FLAGS  =
OPENMP_FLAG = -qopenmp
BLAS_CFLAGS = -DUSE_MKL -I$(MKLROOT)/include
BLAS_LIBS   = -qmkl=parallel
else ifeq ($(USE_NVPL),1)
CC          = gcc
CXX         = g++
FORT        = gfortran
ARCH_FLAGS  = -mcpu=grace
OPENMP_FLAG = -fopenmp
BLAS_CFLAGS = -DUSE_NVPL -I$(NVPLDIR)/include
BLAS_LIBS   = -L$(NVPLDIR)/lib \
	      -lnvpl_lapack_lp64_gomp -lnvpl_blas_lp64_gomp -lgomp
else
CC          = gcc
CXX         = g++
FORT        = gfortran
ARCH_FLAGS  =
OPENMP_FLAG = -fopenmp
BLAS_CFLAGS = -I$(OPENBLAS_INCDIR)
BLAS_LIBS   = -L$(OPENBLAS_LIBDIR) -lopenblas
endif

CPPFLAGS = $(BLAS_CFLAGS)
CFLAGS   = -O3 $(ARCH_FLAGS) $(OPENMP_FLAG)
LDLIBS   = $(BLAS_LIBS) -lm

TARGET = rund3_justQB
SOURCES = driver3_justQB.c \
	  matrix_vector_functions_lapack.c \
	  rank_revealing_algorithms_lapack.c
HEADERS = matrix_vector_functions_lapack.h \
	  rank_revealing_algorithms_lapack.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SOURCES) $(LDLIBS) -o $@

clean:
	$(RM) $(TARGET)
