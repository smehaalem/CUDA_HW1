#include "simulation.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "physics.cuh"

#define BLOCK_X 32
#define BLOCK_Y 8

__constant__ float c_weights[WEIGHTS_COUNT];

__global__ void optimized_heat_step_kernel(
    const float *__restrict__ current,
    float *__restrict__ next,
    int width,
    int height,
    float sx,
    float sy)
{
    __shared__ float s_tile[BLOCK_Y + 2 * STENCIL_RADIUS][BLOCK_X + 2 * STENCIL_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    for (int i = ty; i < BLOCK_Y + 2 * STENCIL_RADIUS; i += blockDim.y)
    {
        for (int j = tx; j < BLOCK_X + 2 * STENCIL_RADIUS; j += blockDim.x)
        {
            int global_x = blockIdx.x * blockDim.x + j - STENCIL_RADIUS;
            int global_y = blockIdx.y * blockDim.y + i - STENCIL_RADIUS;

            s_tile[i][j] = sample_clamped(current, width, height, global_x, global_y);
        }
    }

    __syncthreads();

    if (x >= width || y >= height)
        return;

    float dxs = (float)x - sx;
    float dys = (float)y - sy;
    float dist_sq = dxs * dxs + dys * dys;

    float source = (dist_sq <= SOURCE_RADIUS * SOURCE_RADIUS) ? SOURCE_HEAT : 0.0f;

#pragma unroll
    for (int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++)
    {
#pragma unroll
        for (int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++)
        {
            float val = s_tile[ty + STENCIL_RADIUS + dy][tx + STENCIL_RADIUS + dx];
            float weight = c_weights[(dy + STENCIL_RADIUS) * STENCIL_SIZE + (dx + STENCIL_RADIUS)];

            source += val * weight;
        }
    }

    next[y * width + x] = source;
}

static void swap_buffers(float **a, float **b)
{
    float *tmp = *a;
    *a = *b;
    *b = tmp;
}

void heat_simulate_optimized_init(
    int width,
    int height,
    int num_simulations,
    const float *weights)
{
    CUDA_CHECK(cudaMemcpyToSymbol(
        c_weights,
        weights,
        WEIGHTS_COUNT * sizeof(float)));
}

void heat_simulate_optimized_finalize(void)
{
}

void heat_simulate_optimized(
    float **initial_states,
    const float *weights,
    float **final_states,
    int num_simulations,
    int width,
    int height,
    int start_step,
    int num_steps)
{
    size_t grid_bytes = (size_t)width * (size_t)height * sizeof(float);

    float **d_current = (float **)malloc(num_simulations * sizeof(float *));
    float **d_next = (float **)malloc(num_simulations * sizeof(float *));
    cudaStream_t *streams = (cudaStream_t *)malloc(num_simulations * sizeof(cudaStream_t));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_current[s], grid_bytes));
        CUDA_CHECK(cudaMalloc(&d_next[s], grid_bytes));

        CUDA_CHECK(cudaMemcpyAsync(d_current[s], initial_states[s], grid_bytes, cudaMemcpyHostToDevice, streams[s]));
    }

    for (int step = start_step; step < start_step + num_steps; step++)
    {
        float sx, sy;
        source_position(step, width, height, &sx, &sy);

        for (int s = 0; s < num_simulations; s++)
        {
            optimized_heat_step_kernel<<<grid, block, 0, streams[s]>>>(
                d_current[s],
                d_next[s],
                width,
                height,
                sx,
                sy);

            swap_buffers(&d_current[s], &d_next[s]);
        }
    }

    //  CUDA_CHECK(cudaMemcpyAsync(final_states[s], d_current[s], grid_bytes, cudaMemcpyDeviceToHost, streams[s]));
    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            final_states[s],
            d_current[s],
            grid_bytes,
            cudaMemcpyDeviceToHost,
            streams[s]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaFree(d_current[s]));
        CUDA_CHECK(cudaFree(d_next[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    free(d_current);
    free(d_next);
    free(streams);
}