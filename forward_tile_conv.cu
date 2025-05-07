#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono> // For timing the CPU implementation

// Define matrix dimensions
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKDIM_Z 1
#define UNROLL 1

// Shape structure definition
struct shape {
    int num;
    int depth;
    int height;
    int width;
};

__global__ void conv_forward_opt_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {

  // X[b, c, h+p, w+q] = X[((b * xdims.depth + c) * xdims.height + (h + p)) * xdims.width + (w + q)]
  // W[m, c, p, q] = W[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q]
  // Y[b, m, h, w] = Y[((b * ydims.depth + m) * ydims.height + h) * ydims.width + w]
  const int b     = ydims.num;
  const int wh    = wdims.height;
  const int ww    = wdims.width;
  const int h_in  = ydims.height + wh - 1;
  const int w_in  = ydims.width + ww - 1;
  const int c     = xdims.depth;
  const int m     = ydims.depth;
  const int h_out = h_in - wh + 1;
  const int w_out = w_in - ww + 1;

  int idxXBlockCol = blockIdx.x * m + threadIdx.x;
  int idxM         = blockIdx.y * N + threadIdx.y;

  int numXBlockRows = c * wh * ww;
  int numXBlockCols = h_out * w_out;

  __shared__ float blockMemA[M][K];
  __shared__ float blockMemB[K][N];

  int idxB    = blockIdx.z * BLOCKDIM_Z + threadIdx.z;
  int idxHOut = idxXBlockCol / w_out;
  int idxWOut = idxXBlockCol % w_out;

  if (idxB < b) {
    float PValue[UNROLL];
    for (int unroll = 0; unroll < UNROLL; unroll++) {
      PValue[unroll] = 0.0;
    }
  for (int idx = 0; idx < (numXBlockRows + K - 1) / K; idx++) {
// simplify the matrix indexing
#define Y4d(i3, i2, i1, i0) Y[(i3) * (m * h_out * w_out) + (i2) * (h_out * w_out) + (i1) * (w_out) + i0]
#define W4d(i3, i2, i1, i0) X[(i3) * (c * h_in * w_in) + (i2) * (h_in * w_in) + (i1) * (w_in) + i0]
#define X4d(i3, i2, i1, i0) W[(i3) * (c * wh * ww) + (i2) * (wh * ww) + (i1) * (ww) + i0]

      const int threadIdx1D      = threadIdx.y * BLOCKDIM_X + threadIdx.x;
      const int threadNumInBlock = BLOCKDIM_X * BLOCKDIM_Y;
      int max                    = (M * K + threadNumInBlock - 1) / threadNumInBlock;
      if (((K * N + threadNumInBlock - 1) / threadNumInBlock) > max)
        max = (K * N + threadNumInBlock - 1) / threadNumInBlock;
      for (int idxLoad = 0; idxLoad < max; idxLoad++) {
        int idxYBlockMemALoad = (threadIdx1D + idxLoad * threadNumInBlock) / K;
        int idxXBlockMemALoad = (threadIdx1D + idxLoad * threadNumInBlock) % K;
        int idxYBlockMemBLoad = (threadIdx1D + idxLoad * threadNumInBlock) / N;
        int idxXBlockMemBLoad = (threadIdx1D + idxLoad * threadNumInBlock) % N;

        int idxWColumnToLoad   = idx * K + idxXBlockMemALoad;
        int idxXBlockColToLoad = blockIdx.x * N + idxXBlockMemBLoad;
        int idxXBlockRowToLoad = (idxYBlockMemBLoad + idx * K);

        int idxHOutToLoad = idxXBlockColToLoad / w_out;
        int idxWOutToLoad = idxXBlockColToLoad % w_out;

        int idxCWColumn   = idxWColumnToLoad / (wh * ww);
        int idxMToLoad    = blockIdx.y * M + idxYBlockMemALoad;
        int idxCXBlockRow = idxXBlockRowToLoad / (wh * ww);
        if ((idxCWColumn < c) && (idxMToLoad < m)) {
          int idxKHeightWColumn                           = (idxWColumnToLoad % (wh * ww)) / ww;
          int idxKWidthWColumn                            = (idxWColumnToLoad % (wh * ww)) % ww;
          blockMemA[idxYBlockMemALoad][idxXBlockMemALoad] = X4d(idxMToLoad, idxCWColumn, idxKHeightWColumn, idxKWidthWColumn);
        } else {
          blockMemA[idxYBlockMemALoad][idxXBlockMemALoad] = 0.0;
        }
        if ((idxCXBlockRow < c) && (idxHOutToLoad < h_out)) {
          int idxKHeightXBlockRow = (idxXBlockRowToLoad % (wh * ww)) / ww;
          int idxKWidthXBlockRow  = (idxXBlockRowToLoad % (wh * ww)) % ww;
          blockMemB[idxYBlockMemBLoad][idxXBlockMemBLoad] =
              W4d(idxB, idxCXBlockRow, idxHOutToLoad + idxKHeightXBlockRow, idxWOutToLoad + idxKWidthXBlockRow);
        } else {
          blockMemB[idxYBlockMemBLoad][idxXBlockMemBLoad] = 0.0;
        }
      }
      __syncthreads();
      for (int idx2 = 0; idx2 < K; idx2++) {
        for (int unroll = 0; unroll < UNROLL; unroll++) {
          PValue[unroll] += blockMemA[threadIdx.y + unroll * BLOCKDIM_Y][idx2] * blockMemB[idx2][threadIdx.x];
        }
      }
      __syncthreads();
    }
    if (idxM < M && idxXBlockCol < numXBlockCols) {
      for (int unroll = 0; unroll < UNROLL; unroll++) {
        Y4d(idxB, idxM + unroll * BLOCKDIM_Y, idxHOut, idxWOut) = PValue[unroll];
      }
    }
  }
}


// Baseline matrix calculation (CPU implementation)
void conv_forward_baseline(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {
    for (int b = 0; b < ydims.num; ++b) {
        for (int m = 0; m < ydims.depth; ++m) {
            for (int h = 0; h < ydims.height; ++h) {
                for (int w = 0; w < ydims.width; ++w) {
                    float value = 0.0f;
                    for (int c = 0; c < xdims.depth; ++c) {
                        for (int p = 0; p < wdims.height; ++p) {
                            for (int q = 0; q < wdims.width; ++q) {
                                int h_in = h + p;
                                int w_in = w + q;
                                value += X[((b * xdims.depth + c) * xdims.height + h_in) * xdims.width + w_in] *
                                         W[((m * wdims.depth + c) * wdims.height + p) * wdims.width + q];
                            }
                        }
                    }
                    Y[((b * ydims.depth + m) * ydims.height + h) * ydims.width + w] = value;
                }
            }
        }
    }
}

// Example to test the kernel and baseline
int main() {
    // Define input dimensions
    shape xdims = {1, 3, 5, 5};  // Batch size = 1, Depth = 3, Height = 5, Width = 5
    shape wdims = {2, 3, 3, 3};  // Filters = 2, Depth = 3, Height = 3, Width = 3
    shape ydims = {1, 2, 3, 3};  // Batch size = 1, Depth = 2, Height = 3, Width = 3

    // Allocate host memory
    std::vector<float> X(xdims.num * xdims.depth * xdims.height * xdims.width, 1.0f);  // Input initialized to 1
    std::vector<float> W(wdims.num * wdims.depth * wdims.height * wdims.width, 1.0f);  // Weights initialized to 1
    std::vector<float> Y_baseline(ydims.num * ydims.depth * ydims.height * ydims.width, 0.0f);
    std::vector<float> Y_device(ydims.num * ydims.depth * ydims.height * ydims.width, 0.0f);

    // Allocate device memory
    float *d_X, *d_W, *d_Y;
    cudaMalloc(&d_X, X.size() * sizeof(float));
    cudaMalloc(&d_W, W.size() * sizeof(float));
    cudaMalloc(&d_Y, Y_device.size() * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Measure execution time for GPU kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dim3 blockDim(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
    dim3 gridDim((ydims.width + BLOCKDIM_X - 1) / BLOCKDIM_X, (ydims.depth + BLOCKDIM_Y - 1) / BLOCKDIM_Y, ydims.num);
    conv_forward_opt_kernel<<<gridDim, blockDim>>>(d_X, xdims, d_W, wdims, d_Y, ydims);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy result back to host
    cudaMemcpy(Y_device.data(), d_Y, Y_device.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure execution time for baseline calculation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    conv_forward_baseline(X.data(), xdims, W.data(), wdims, Y_baseline.data(), ydims);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // Compare results
    std::cout << "GPU Time (ms): " << gpu_time << std::endl;
    std::cout << "CPU Time (ms): " << cpu_time.count() << std::endl;

    for (int i = 0; i < Y_baseline.size(); ++i) {
        std::cout << "Y_baseline[" << i << "] = " << Y_baseline[i] << ", Y_device[" << i << "] = " << Y_device[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);

    return 0;
}
