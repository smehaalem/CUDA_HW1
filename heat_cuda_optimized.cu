#include "simulation.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "physics.cuh"

/*
 * 32 in X keeps memory accesses coalesced.
 * 16 in Y reduces the number of blocks and halo overhead compared to 32x8.
 */
#define BLOCK_X 32
#define BLOCK_Y 16

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
    int timestep)
{
    /*
     * Shared-memory tile:
     * BLOCK area + halo on all sides.
     * The +1 in the X dimension helps reduce shared-memory bank conflicts.
     */
    __shared__ float s_tile
        [BLOCK_Y + 2 * STENCIL_RADIUS]
        [BLOCK_X + 2 * STENCIL_RADIUS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_origin_x = blockIdx.x * BLOCK_X;
    const int block_origin_y = blockIdx.y * BLOCK_Y;

    const int x = block_origin_x + tx;
    const int y = block_origin_y + ty;

    /*
     * Most blocks are interior blocks.
     * For them, the entire tile including halo is inside the grid,
     * so direct global-memory loads are equivalent to sample_clamped.
     *
     * Boundary blocks still use sample_clamped exactly like the reference,
     * preserving correctness.
     */
    const bool interior_block =
        block_origin_x >= STENCIL_RADIUS &&
        block_origin_y >= STENCIL_RADIUS &&
        block_origin_x + BLOCK_X + STENCIL_RADIUS <= width &&
        block_origin_y + BLOCK_Y + STENCIL_RADIUS <= height;

    /*
     * Load tile + halo into shared memory.
     * Threads cooperatively load more elements than output cells because of halo.
     */
    for (int local_y = ty;
         local_y < BLOCK_Y + 2 * STENCIL_RADIUS;
         local_y += BLOCK_Y)
    {
        const int gy = block_origin_y + local_y - STENCIL_RADIUS;

        for (int local_x = tx;
             local_x < BLOCK_X + 2 * STENCIL_RADIUS;
             local_x += BLOCK_X)
        {
            const int gx = block_origin_x + local_x - STENCIL_RADIUS;

            if (interior_block)
            {
                s_tile[local_y][local_x] =
                    current[gy * width + gx];
            }
            else
            {
                s_tile[local_y][local_x] =
                    sample_clamped(current, width, height, gx, gy);
            }
        }
    }

    /*
     * Required because all threads must finish loading the shared tile
     * before any thread starts reading stencil neighbors from it.
     */
    __syncthreads();

    if (x >= width || y >= height)
    {
        return;
    }

    /*
     * Keep the exact reference heat source for correctness.
     * This avoids numerical differences from manually reimplementing it.
     */
    float source = heat_source(timestep, x, y, width, height);

#pragma unroll
    for (int dy = -STENCIL_RADIUS; dy <= STENCIL_RADIUS; dy++)
    {
#pragma unroll
        for (int dx = -STENCIL_RADIUS; dx <= STENCIL_RADIUS; dx++)
        {
            const float val =
                s_tile[ty + STENCIL_RADIUS + dy]
                      [tx + STENCIL_RADIUS + dx];

            const float weight =
                c_weights[(dy + STENCIL_RADIUS) * STENCIL_SIZE
                        + (dx + STENCIL_RADIUS)];

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

    if (g_d_current == NULL || g_d_next == NULL || g_streams == NULL)
    {
        fprintf(stderr, "Host malloc failed in heat_simulate_optimized_init\n");
        exit(1);
    }

    for (int s = 0; s < num_simulations; s++)
    {
        CUDA_CHECK(cudaStreamCreate(&g_streams[s]));
        CUDA_CHECK(cudaMalloc(&g_d_current[s], g_grid_bytes));
        CUDA_CHECK(cudaMalloc(&g_d_next[s], g_grid_bytes));
    }
}

void heat_simulate_optimized_finalize(void)
{
    if (g_d_current != NULL && g_d_next != NULL && g_streams != NULL)
    {
        for (int s = 0; s < g_num_simulations; s++)
        {
            CUDA_CHECK(cudaFree(g_d_current[s]));
            CUDA_CHECK(cudaFree(g_d_next[s]));
            CUDA_CHECK(cudaStreamDestroy(g_streams[s]));
        }
    }

    free(g_d_current);
    free(g_d_next);
    free(g_streams);

    g_d_current = NULL;
    g_d_next = NULL;
    g_streams = NULL;
    g_num_simulations = 0;
    g_grid_bytes = 0;
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
    const size_t grid_bytes =
        (size_t)width * (size_t)height * sizeof(float);

    dim3 block(BLOCK_X, BLOCK_Y);

    dim3 grid(
        (width + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y);

    /*
     * Copy all simulations to the GPU.
     * We keep streams because they were explicitly covered in the CUDA material
     * and the previous version already used them correctly.
     */
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
        for (int s = 0; s < num_simulations; s++)
        {
            optimized_heat_step_kernel<<<grid, block, 0, g_streams[s]>>>(
                g_d_current[s],
                g_d_next[s],
                width,
                height,
                step);

            swap_buffers(&g_d_current[s], &g_d_next[s]);
        }
    }

    CUDA_CHECK(cudaGetLastError());

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