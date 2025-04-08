#include "cal_anm.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
// #include "error.cuh"
#include <iostream>
#include <unistd.h>


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSolver error in " << __FILE__ << " at line " << __LINE__ << ": " << err << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}



void eigh_gpu(double * eigenvalues, double * eigenvectors, double * matrix, int * eigvals, int n, int rank, int size){

    int gpu_device = rank % 4;
    CHECK_CUDA(cudaSetDevice(gpu_device));

    cusolverDnHandle_t cusolver_handle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));

    int lda = n;
    double *d_eigenvalues = nullptr, *d_matrix = nullptr, *d_work = nullptr;
    int *d_info = nullptr, info = 0, lwork = 0;

    double *eigenvectors_buffer = (double *)malloc(n * n * sizeof(double));

    CHECK_CUDA(cudaMalloc((void**)&d_matrix, n * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_eigenvalues, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_matrix, matrix, n * n * sizeof(double), cudaMemcpyHostToDevice));

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // Compute eigenvalues & eigenvectors
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolver_handle, jobz, uplo, n, d_matrix, lda, d_eigenvalues, &lwork));
    lwork = static_cast<int>(lwork * 1.5);  

    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(double) * lwork));

    // Compute eigenvalues
    CHECK_CUSOLVER(cusolverDnDsyevd(cusolver_handle, jobz, uplo, n, d_matrix, lda, d_eigenvalues, d_work, lwork, d_info));

    // 确保计算完成
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "cuSolver failed on rank " << rank << ", info = " << info << std::endl;
        exit(EXIT_FAILURE);
    }

    int start = eigvals[0], end = eigvals[1], length = end - start + 1;
    CHECK_CUDA(cudaMemcpy(eigenvectors_buffer, d_matrix, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(eigenvalues, d_eigenvalues + start, length * sizeof(double), cudaMemcpyDeviceToHost));

    // 释放资源
    CHECK_CUDA(cudaFree(d_matrix));
    CHECK_CUDA(cudaFree(d_eigenvalues));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolver_handle));
    CHECK_CUDA(cudaDeviceReset());

    // 重新整理 eigenvectors
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < n; j++) {
            eigenvectors[j * length + i] = eigenvectors_buffer[i * n + j];
        }
    }
    free(eigenvectors_buffer);
    

}





