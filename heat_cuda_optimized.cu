#include "simulation.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "physics.cuh"
#define BLOCK_X 32
#define BLOCK_Y 8

__constant__ float c_weights[WEIGHTS_COUNT];
static float **g_d_current = NULL;
static float **g_d_next = NULL;
static cudaStream_t *g_streams = NULL;
static int g_num_simulations = 0;
static size_t g_grid_bytes = 0;
__global__ void optimized_heat_step_kernel(
    const float *__restrict__ current,
    float *__restrict__ next,
    int width,
    int height,
    float sx,
    float sy)
{
    __shared__ float s_tile[BLOCK_Y + 2 * STENCIL_RADIUS][BLOCK_X + 2 * STENCIL_RADIUS + 1];
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
    g_num_simulations = num_simulations;
    g_grid_bytes = (size_t)width * (size_t)height * sizeof(float);
    g_d_current = (float **)malloc(num_simulations * sizeof(float *));
    g_d_next = (float **)malloc(num_simulations * sizeof(float *));
    g_streams = (cudaStream_t *)malloc(num_simulations * sizeof(cudaStream_t));
    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaStreamCreate(&g_streams[s]));
        CUDA_CHECK(cudaMalloc(&g_d_current[s], g_grid_bytes));
        CUDA_CHECK(cudaMalloc(&g_d_next[s], g_grid_bytes));
    }
}
void heat_simulate_optimized_finalize(void)
{
    for (int s = 0; s < g_num_simulations; s++)
    {
        CUDA_CHECK(cudaFree(g_d_current[s]));
        CUDA_CHECK(cudaFree(g_d_next[s]));
        CUDA_CHECK(cudaStreamDestroy(g_streams[s]));
    }
    free(g_d_current);
    free(g_d_next);
    free(g_streams);
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
    size_t grid_bytes =
        (size_t)width * (size_t)height * sizeof(float);
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            g_d_current[s],
            initial_states[s],
            grid_bytes,
            cudaMemcpyHostToDevice,
            g_streams[s]));
    }

    for (int step = start_step;
         step < start_step + num_steps;
         step++)
    {
        float sx, sy;
        source_position(
            step,
            width,
            height,
            &sx,
            &sy);
        for (int s = 0; s < num_simulations; s++)
        {
            optimized_heat_step_kernel<<<
                grid,
                block,
                0,
                g_streams[s]>>>(
                g_d_current[s],
                g_d_next[s],
                width,
                height,
                sx,
                sy);
            swap_buffers(
                &g_d_current[s],
                &g_d_next[s]);
        }
    }

    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            final_states[s],
            g_d_current[s],
            grid_bytes,
            cudaMemcpyDeviceToHost,
            g_streams[s]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}