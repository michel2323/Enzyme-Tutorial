#include <stdio.h>

// Kernel for computing dot product of a vector with itself
void __device__ dotProductKernel_impl(double* x, double* result, int n) {
    // Shared memory for partial sums within a block
    __shared__ double sdata[256];  // Adjust size as needed

    // Each thread computes partial sum for strided elements
    double thread_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process multiple elements per thread (grid-stride loop)
    for (int i = idx; i < n; i += stride) {
        thread_sum += x[i] * x[i];
    }

    // Store in shared memory
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction within block
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // First thread in block writes result to global memory
    if (threadIdx.x == 0) {
        atomicAdd(result, sdata[0]);
    }
}

typedef void (*f_ptr)(double*, double*, int);

extern void __device__ __enzyme_autodiff(f_ptr,
    int, double*, double*,
    int, double*, double*,
    int, int
);

void __global__ dotProductKernel(double* x, double* result, int n) {
    dotProductKernel_impl(x, result, n);
}

int __device__ enzyme_dup;
int __device__ enzyme_out;
int __device__ enzyme_const;

void __global__ dotProductKernel_grad(double* x, double* d_x, double* result, double* d_result, int n) {

    __enzyme_autodiff(dotProductKernel_impl,
        enzyme_dup, x, d_x,
        enzyme_dup, result, d_result,
        enzyme_const, n);

}

int main() {
    const int n = 16;  // Vector size
    double *h_x, *h_result;  // Host data
    double *dh_x, *dh_result;  // Host data
    double *d_x, *d_result;  // Device data
    double *dd_x, *dd_result;  // Device data

    // Allocate host memory
    h_x = (double*)malloc(n * sizeof(double));
    dh_x = (double*)malloc(n * sizeof(double));
    h_result = (double*)malloc(sizeof(double));
    dh_result = (double*)malloc(sizeof(double));

    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_x[i] = i+1;  // Example values
        dh_x[i] = 0.0;
    }
    *h_result = 0.0;
    *dh_result = 1.0;

    // Allocate device memory
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&dd_x, n * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));
    cudaMalloc(&dd_result, sizeof(double));

    // Copy data to device
    cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_x, dh_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_result, dh_result, sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
#ifdef FORWARD
    dotProductKernel<<<numBlocks, blockSize>>>(d_x, d_result, n);
#else
    dotProductKernel_grad<<<numBlocks, blockSize>>>(d_x, dd_x, d_result, dd_result, n);
#endif

    // Copy result back to host
    cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dh_x, dd_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    printf("dot(x,x) = %f\n", *h_result);
    for(int i = 0; i < n; i++) {
        printf("dh_x[%d] = %f\n", i, dh_x[i]);
    }

    // Free memory
    free(h_x);
    free(h_result);
    free(dh_x);
    free(dh_result);
    cudaFree(d_x);
    cudaFree(dd_x);
    cudaFree(d_result);
    cudaFree(dd_result);

    return 0;
}