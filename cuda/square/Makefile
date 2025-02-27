# Environment Variables
CUDA_PATH = /usr/local/cuda
ENZYME_PATH = /disk/mschanen/git/Enzyme/build/Enzyme/ClangEnzyme-21.so
CC = clang++
SM_VERSION  = 70
GCC_LIB_PATH = /usr//lib/gcc/x86_64-linux-gnu/11

# Complete C++ include paths
CXX_INCLUDES = -isystem /usr/include/c++/11 \
               -isystem /usr/include/x86_64-linux-gnu/c++/11 \
               -isystem /usr/include/c++/11/backward \
               -isystem /usr/lib/gcc/x86_64-linux-gnu/11/include \
               -isystem /usr/local/include \
               -isystem /usr/include/x86_64-linux-gnu \
               -isystem /usr/include \
               -isystem $(CUDA_PATH)/include


CFLAGS = -x cuda $(CXX_INCLUDES) -fno-vectorize -O2 -fno-unroll-loops -fPIC --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_$(SM_VERSION)\
		--no-cuda-version-check -Xclang -load -Xclang $(ENZYME_PATH)
LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(GCC_LIB_PATH) -lcudart_static -lstdc++

all: cuda_square cuda_dot

cuda_square: cuda_square.cu
	$(CC) $(CFLAGS) cuda_square.cu -o cuda_square $(LDFLAGS)
cuda_dot: cuda_dot.cu
	$(CC) $(CFLAGS) cuda_dot.cu -o cuda_dot $(LDFLAGS)

clean:
	rm -f cuda_square cuda_dot