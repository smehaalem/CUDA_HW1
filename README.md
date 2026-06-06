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

We started from the reference kernel and improved it in small steps, running the benchmark after each change to make sure it still printed PASS.

The first thing was shared memory. In the reference every cell reads all 169 neighbors straight from global memory, and neighbor cells read mostly the same values, so the same data gets read over and over. So each block loads its part of the grid plus a halo into shared memory once, does __syncthreads(), and then runs the stencil from shared memory. This is basically the 2D version of the stencil example from the tutorial, and it helped a lot.

We also put the weight matrix in constant memory, since it never changes and all threads use the same values. When the whole warp reads the same weight it gets broadcast, so it's almost free.

The biggest win was how we launch the kernel. At first we launched one kernel per simulation, which is 300 steps * 10 sims = 3000 launches. The grid is only 128x128 so each kernel does almost no work, and most of the time was just launch overhead. We put all 10 simulations in one buffer and used the z dimension of the grid (blockIdx.z) so one launch advances all of them at once. That dropped it to 300 launches and roughly cut the time in half.

A smaller one: the source position uses cos/sin but only depends on the timestep, not on x/y, so every thread was computing the same thing. We let one thread compute it into shared memory and the rest of the block reuse it.

Last, once the slow part was the shared-memory reads in the stencil loop (169 per cell), we made each thread compute two pixels stacked on top of each other instead of one. Both use the same loaded tile, so the reads are shared between them. We kept the same dy/dx loop order as the reference so the result stays exactly the same.

Things we tried that didn't help: we also tested 4 pixels per thread, but it was slower - with fewer threads the GPU has less to overlap, so we went back to 2. The interior-block fast path (skipping the boundary check for blocks fully inside the grid) also didn't really change anything, we just left it in.

In the end we got PASS, around 5 ms vs about 41 ms for the reference (it moves a bit depending on how busy the server is).

Reference we used: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

---

### Submission

- Submission is in **pairs**. Submit a **zip file** named `<studentID1>_<studentID2>.zip`.
- The zip must contain a single folder with:
  - This `README.md`, with **A1** filled in
  - `**heat_cuda_optimized.cu`** only
- Do **not** include the `Makefile`, other source files, compiled binaries, `frames/`, or any other files.

