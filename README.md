# 02360509 - Advanced topics in hardware accelerators for deep learning

## HW 1 - Heat Diffusion on GPU - 20% of final grade

This assignment simulates **2D heat diffusion** on a GPU: each timestep applies a large **weighted stencil** over the temperature grid that includes a moving heat source. The weights and a **reference** CUDA implementation are provided. Your job is to make it run as **fast** as possible while staying **correct**.

The computation kernel itself is simple; the interesting part is **optimization**. You may use anything covered in the course and anything you find online, as long as you stick to **native CUDA** (kernels, memory types, streams, etc.). Do **not** use external libraries or tools that optimize or tune for you (e.g. autotuners, vendor BLAS, third-party GPU frameworks). You are expected to write the optimizations yourself.

**Grading (within this 20%):**


| Component                    | Weight      | Criteria                                                                                                                                                              |
| ---------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Correctness & speed**      | 80% of HW 1 | Output must match the reference (`PASS` in `benchmark`). For any points, your optimized kernel must also be **faster than the reference** on the course test machine. |
| **Optimization competition** | 20% of HW 1 | Ranked against other pairs on the course server - fastest correct submissions score highest.                                                                          |


### Building & Running

A `Makefile` is provided:

```bash
make          # build benchmark and plot
make clean    # remove binaries and frames/
```

`**benchmark**` - correctness check and timing (reference vs optimized):

```bash
./benchmark
```

Prints `**PASS**` or `**FAIL**` by comparing your output to the reference across multiple simulations, then mean / min / max / stddev timing for both implementations. If you see `**PASS**`, your result is numerically correct. For grading credit on the 80% portion, your optimized time must also beat the reference on the lambda server.

`**plot**` - writes BMP frames to `frames/`. Handy for debugging, and as a bonus you get to watch the heat spread and the source move around.

```bash
./plot reference
./plot optimized
```

### References

You may use ideas from lectures, CUDA samples, blogs, or papers. Do **not** copy code verbatim. If you adapt material from outside the course, cite it in your dry answer and include **links** to the sources.

### Environment

You are welcome to develop on any machine with a CUDA-compatible GPU or using Google Colab. However, your submission will be tested on the **lambda** server using an **NVIDIA GeForce RTX 2080 Ti**. Verify `**PASS`** and your speedup there before submitting.

For this assignment that matters especially: an optimization that helps on one GPU (e.g. your laptop or Colab) may not give the same speedup on another architecture, and can even hurt. Treat the RTX 2080 Ti on lambda as the machine that counts for grading and the competition.

### Dry Questions

Write your answer under the **A1.** label. A short but concrete write-up is enough (what you tried, what worked, what did not, and why you think so). We recommend updating this section continuously while you optimize.

---

**Q1.** Briefly describe your optimization work: what approaches did you try, what improved performance, what didn't, and your reasoning. If you used external references (outside course material), list them with links.

**A1.**

We improved the reference kernel gradually, checking PASS and the time after each change.

- **Shared memory.** Each cell in the 13x13 stencil is accessed by many neighbors, so the reference reads them all from global memory. We loaded a tile of the grid plus a halo into shared memory once per block and ran the stencil from there, using __syncthreads() after loading. This is essentially the 2D version of the stencil example from the tutorial. It was a significant improvement.

- **Constant memory.** The weights are read-only and every thread uses the same values, so we moved them to constant memory. When all threads in a warp access the same weight, it is broadcast in a single access, which worked well for us.

- **One launch for all simulations (biggest win).** Initially, we launched one kernel per simulation, which totaled 300 steps times 10 simulations, making 3000 launches. Since the grid is only 128x128, each kernel is small, and the time was mostly lost in launch overhead rather than actual work. We combined all 10 simulations into one buffer and utilized the z dimension of the grid (blockIdx.z) so one launch would advance all simulations together. This reduced the launches to only 300 and roughly halved the time.

- **Source once per block.** heat_source uses cosf/sinf through source_position, but the source position only depends on the timestep, not x or y. As a result, every thread was recomputing the same value. We let one thread calculate it and store it in shared memory for the rest of the block to reuse.

- **Two pixels per thread.** After the above changes, the slowest part was the shared-memory reads in the stencil loop, which were 169 per cell. We made each thread compute two stacked pixels using the same loaded tile, allowing those reads to be shared between the two pixels. We maintained the same dy/dx loop order as the reference to keep the values consistent. This adjustment helped us go under 5 ms.

What didn’t really help:
- We also tried four pixels per thread with fewer threads per block. This approach actually slowed down performance. With fewer threads, the GPU had less work to overlap, meaning the loss of parallelism outweighed the benefits of extra reuse. So we decided to stick with two pixels per thread.
- The interior-block fast path, which skips the boundary clamp for blocks fully inside the grid, made almost no difference once the kernel was limited by the stencil reads rather than the load. We kept it since it did not negatively impact performance.

Final results on the lambda RTX 2080 Ti: PASS, about 4.7 ms compared to around 41 ms for the reference.

Reference we looked at: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

---

### Submission

- Submission is in **pairs**. Submit a **zip file** named `<studentID1>_<studentID2>.zip`.
- The zip must contain a single folder with:
  - This `README.md`, with **A1** filled in
  - `**heat_cuda_optimized.cu`** only
- Do **not** include the `Makefile`, other source files, compiled binaries, `frames/`, or any other files.

