# CUDA Patterns Reference

## Contents

- [Tiled Matrix Multiplication](#tiled-matrix-multiplication)
- [Parallel Reduction Tree](#parallel-reduction-tree)
- [Stream Overlap Pipeline](#stream-overlap-pipeline)

## Tiled Matrix Multiplication

Computes C = A * B where A is MxK, B is KxN, C is MxN.
Each block loads a TILE_SIZE x TILE_SIZE sub-matrix into shared memory,
reducing global memory accesses from O(K) per element to O(K / TILE_SIZE).

```cuda
constexpr int TILE = 32;

__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1]; // +1 padding avoids bank conflicts

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Coalesced load from global to shared (row-major layout)
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Compute partial dot product from the tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Launch: one block per TILE x TILE output tile
void launch_matmul(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_KERNEL();
}
```

**Why it works:** Each element of C requires K multiply-adds. Without tiling,
every thread reads K floats from global memory. With TILE=32, each global
load is reused 32 times from shared memory, yielding a 32x reduction in
global memory traffic. The `+1` padding on `Bs` prevents shared memory
bank conflicts when threads in a warp access the same column.

## Parallel Reduction Tree

Reduces an array of N floats to a single sum. Uses a two-phase approach:
per-block reduction into partial sums, then a final reduction of partials.

```cuda
// Warp-level reduction using shuffle -- no shared memory, no __syncthreads
__device__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction: warp shuffle + shared memory for cross-warp
__global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float warp_results[32];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread loads two elements (reduces idle threads in first step)
    float sum = 0.0f;
    if (idx < n)               sum += input[idx];
    if (idx + blockDim.x < n)  sum += input[idx + blockDim.x];

    // Intra-warp reduction
    sum = warp_reduce(sum);

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    if (lane == 0) {
        warp_results[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces the per-warp results
    int num_warps = blockDim.x / warpSize;
    sum = (tid < num_warps) ? warp_results[tid] : 0.0f;
    if (warp_id == 0) {
        sum = warp_reduce(sum);
    }

    // Block result
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// Host-side: two-pass reduction
float reduce(const float* d_input, int n) {
    int block_size = 256;
    int grid_size = (n + block_size * 2 - 1) / (block_size * 2);

    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

    // Pass 1: N elements -> grid_size partial sums
    reduce_sum<<<grid_size, block_size>>>(d_input, d_partial, n);
    CUDA_CHECK_KERNEL();

    // Pass 2: grid_size partial sums -> 1 final sum
    // For small grid_size, a single block suffices
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    reduce_sum<<<1, block_size>>>(d_partial, d_result, grid_size);
    CUDA_CHECK_KERNEL();

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float),
                           cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}
```

**Performance notes:**
- The "load two elements" trick doubles useful work per thread and
  eliminates the idle-thread waste in the first reduction step.
- `__shfl_down_sync` is faster than shared memory for intra-warp
  communication (no memory access, no bank conflicts).
- For very large arrays (>100M elements), add a grid-stride accumulation
  loop before the warp reduction to reduce grid_size further.

## Stream Overlap Pipeline

Overlaps host-to-device transfer, kernel execution, and device-to-host
transfer using multiple CUDA streams. Requires pinned host memory.

```cuda
// Pipeline: divide work into chunks, overlap transfer + compute
void stream_pipeline(
    const float* h_input,    // Pinned host memory (cudaMallocHost)
    float* h_output,         // Pinned host memory
    int total_elements
) {
    constexpr int NUM_STREAMS = 4;
    int chunk_size = (total_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    size_t chunk_bytes = chunk_size * sizeof(float);

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate device memory for all chunks
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_elements * sizeof(float)));

    // Issue async operations per stream
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * chunk_size;
        int count = min(chunk_size, total_elements - offset);
        size_t bytes = count * sizeof(float);

        // Stage 1: Async copy host -> device
        CUDA_CHECK(cudaMemcpyAsync(
            d_input + offset, h_input + offset,
            bytes, cudaMemcpyHostToDevice, streams[i]));

        // Stage 2: Kernel execution
        int block_size = 256;
        int grid_size = (count + block_size - 1) / block_size;
        process_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            d_input + offset, d_output + offset, count);

        // Stage 3: Async copy device -> host
        CUDA_CHECK(cudaMemcpyAsync(
            h_output + offset, d_output + offset,
            bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all streams to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Caller must use pinned memory for async transfers to actually overlap
void run_pipeline() {
    int n = 10000000;
    size_t bytes = n * sizeof(float);

    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in,  bytes));  // Pinned
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));   // Pinned

    // Initialize input ...
    for (int i = 0; i < n; ++i) h_in[i] = (float)i;

    stream_pipeline(h_in, h_out, n);

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
}
```

**Timeline with 4 streams (ideal overlap):**

```
Stream 0: [H2D][Kernel][D2H]
Stream 1:      [H2D][Kernel][D2H]
Stream 2:           [H2D][Kernel][D2H]
Stream 3:                [H2D][Kernel][D2H]
```

**Key requirements for overlap:**
- Host memory MUST be pinned (`cudaMallocHost`); pageable memory forces synchronous transfers
- The GPU must have a copy engine separate from the compute engine (all modern GPUs do)
- Use `cudaMemcpyAsync` -- synchronous `cudaMemcpy` blocks the host thread
- Chunk sizes should be large enough to saturate PCIe bandwidth (~64 KB minimum, 1-4 MB recommended)
- Verify overlap with `nsys profile` -- look for concurrent copy and compute on the timeline
