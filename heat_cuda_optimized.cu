#include "simulation.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "physics.cuh"
__constant__ float c_weights[WEIGHTS_COUNT];
__global__ void optimized_heat_step_kernel(
    const float *current,
    float *next,
    int width,
    int height,
    int timestep,
    float sx,
    float sy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

   // float source = heat_source(timestep, x, y, width, height);
    float dxs = (float)x - sx;
    float dys = (float)y - sy;

    float dist_sq = dxs*dxs + dys*dys;

    float source =
        (dist_sq <= SOURCE_RADIUS*SOURCE_RADIUS)
        ? SOURCE_HEAT
        : 0.0f;
    #pragma unroll
    for (int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++) {
        #pragma unroll
        for (int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++) {

            source += sample_clamped(
                          current,
                          width,
                          height,
                          x + dx,
                          y + dy)
                    * c_weights[(dy + STENCIL_RADIUS) * STENCIL_SIZE +
                                (dx + STENCIL_RADIUS)];
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
    float *d_current = NULL;
    float *d_next = NULL;
    size_t grid_bytes = (size_t)width * (size_t)height * sizeof(float);

    for (int s = 0; s < num_simulations; s++) {
        CUDA_CHECK(cudaMalloc(&d_current, grid_bytes));
        CUDA_CHECK(cudaMalloc(&d_next, grid_bytes));
        CUDA_CHECK(cudaMemcpy(d_current, initial_states[s], grid_bytes, cudaMemcpyHostToDevice));

        dim3 block(32,8);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        for (int step = start_step; step < start_step + num_steps; step++) {
            float sx, sy;
            source_position(step, width, height, &sx, &sy);
            optimized_heat_step_kernel<<<grid, block>>>(
                d_current, d_next, width, height, step, sx, sy);
            CUDA_CHECK(cudaGetLastError());
            swap_buffers(&d_current, &d_next);
        }

        CUDA_CHECK(cudaMemcpy(final_states[s], d_current, grid_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
    }
}

