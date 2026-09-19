Problem (benchmarking_script):
    b)  foward = 19.99ms
        forward+backward = 61.74 ms
        full_training_step = 69.92 ms
        bacward addition is around 41.75 ms
        optimizer_step addition is around 8.18 ms
        full_training_step is 3.5x slower than one forward
    c)  w=0: 73.19 ms
        w=1: 24.13 ms
        w=2: 20.30 ms
        w=3: 20.24 ms
        cold start overhead during the first use particular CUDA operations, CUDA libraries create internal sate and buffers, PyTorch allocates memory for activations for the first time for activation, GPU ramps up from a low power state to its operating frequency. First warm-up sometimes in insufficient , because not all initialization and stabilization completes in time with first forward