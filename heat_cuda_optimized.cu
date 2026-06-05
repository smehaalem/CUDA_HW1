#include "simulation.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "physics.cuh"

#define BLOCK_X 32
#define BLOCK_Y 16
#define ROWS_PER_THREAD 2
#define OUT_Y (BLOCK_Y * ROWS_PER_THREAD)

__constant__ float c_weights[WEIGHTS_COUNT];

static float *d_current = NULL;
static float *d_next = NULL;
static int sim_size = 0;

__global__ void optimized_heat_step_kernel(
    const float *current,
    float *next,
    int width,
    int height,
    int timestep,
    int sim_size)
{
    __shared__ float tile[OUT_Y + 2 * STENCIL_RADIUS][BLOCK_X + 2 * STENCIL_RADIUS];
    __shared__ float source_x;
    __shared__ float source_y;

    const float *cur = current + blockIdx.z * sim_size;
    float *nxt = next + blockIdx.z * sim_size;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int origin_x = blockIdx.x * BLOCK_X;
    int origin_y = blockIdx.y * OUT_Y;

    int x = origin_x + tx;
    int row0 = ty * ROWS_PER_THREAD;
    int y0 = origin_y + row0;
    int y1 = y0 + 1;

    bool inside = origin_x >= STENCIL_RADIUS &&
                  origin_y >= STENCIL_RADIUS &&
                  origin_x + BLOCK_X + STENCIL_RADIUS <= width &&
                  origin_y + OUT_Y + STENCIL_RADIUS <= height;

    for (int ly = ty; ly < OUT_Y + 2 * STENCIL_RADIUS; ly += BLOCK_Y)
    {
        for (int lx = tx; lx < BLOCK_X + 2 * STENCIL_RADIUS; lx += BLOCK_X)
        {
            int gx = origin_x + lx - STENCIL_RADIUS;
            int gy = origin_y + ly - STENCIL_RADIUS;
            if (inside)
            {
                tile[ly][lx] = cur[gy * width + gx];
            }
            else
            {
                tile[ly][lx] = sample_clamped(cur, width, height, gx, gy);
            }
        }
    }

    if (tx == 0 && ty == 0)
    {
        source_position(timestep, width, height, &source_x, &source_y);
    }

    __syncthreads();

    float fx = (float)x - source_x;
    float fy0 = (float)y0 - source_y;
    float fy1 = (float)y1 - source_y;

    float r0 = (fx * fx + fy0 * fy0 <= SOURCE_RADIUS * SOURCE_RADIUS) ? SOURCE_HEAT : 0.0f;
    float r1 = (fx * fx + fy1 * fy1 <= SOURCE_RADIUS * SOURCE_RADIUS) ? SOURCE_HEAT : 0.0f;

    for (int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++)
    {
        for (int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++)
        {
            float w = c_weights[(dy + STENCIL_RADIUS) * STENCIL_SIZE + (dx + STENCIL_RADIUS)];
            int col = tx + STENCIL_RADIUS + dx;
            r0 += tile[row0 + STENCIL_RADIUS + dy][col] * w;
            r1 += tile[row0 + 1 + STENCIL_RADIUS + dy][col] * w;
        }
    }

    if (x < width)
    {
        if (y0 < height)
        {
            nxt[y0 * width + x] = r0;
        }
        if (y1 < height)
        {
            nxt[y1 * width + x] = r1;
        }
    }
}

void heat_simulate_optimized_init(
    int width,
    int height,
    int num_simulations,
    const float *weights)
{
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, weights, WEIGHTS_COUNT * sizeof(float)));

    sim_size = width * height;

    CUDA_CHECK(cudaMalloc(&d_current, (size_t)num_simulations * sim_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next, (size_t)num_simulations * sim_size * sizeof(float)));
}

void heat_simulate_optimized_finalize(void)
{
    cudaFree(d_current);
    cudaFree(d_next);
    d_current = NULL;
    d_next = NULL;
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
    int bytes = sim_size * sizeof(float);

    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaMemcpy(d_current + s * sim_size, initial_states[s], bytes, cudaMemcpyHostToDevice));
    }

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((width + BLOCK_X - 1) / BLOCK_X,
              (height + OUT_Y - 1) / OUT_Y,
              num_simulations);

    for (int step = start_step; step < start_step + num_steps; step++)
    {
        optimized_heat_step_kernel<<<grid, block>>>(d_current, d_next, width, height, step, sim_size);

        float *tmp = d_current;
        d_current = d_next;
        d_next = tmp;
    }

    CUDA_CHECK(cudaGetLastError());

    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaMemcpy(final_states[s], d_current + s * sim_size, bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}
