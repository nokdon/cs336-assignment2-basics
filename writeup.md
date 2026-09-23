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
Problem (nsys_profile):
    a)  the forward NVTX range lasted 19.62ms, while measured_step including GPU synchronization lasted 19.69ms. This is close to earilier timeit results which is 20ms
    b) During forward "sm90_xmma_gemm_…_tn_…_tilesize64x256x32_…" had the highest
  cumulative GPU time: 60 invocations, 1.112ms. With backward included, the leading change to "vectorized_elementwise_kernel<…MulFunctor…>", with 196 invocations|3.452ms
    c) elementwise muls, exponention, masking through where, maximum and sum reductions
    d)The fraction of cumulative GPU kernel time spent in GEMM kernels ecreased from 33.1% -> 27.1% even though absolute exectuion time increased. Meanwhile, the share of vectorized elementwise addition kernel increased from 3.1% to 10.2%
    e) Softmax took 2.147 ms, compared to 0.708 ms for two attention matrix mul combined(softmax x3 slower). Per head seq and layer the two attention matmuls require 67.11 million FLOPs, while softmax 1.31 million (51x lower). SHowing that operation counts alone don't predict runtime

Problem (mixed_precision_accumulation):
    In the first tow cases, float32 gives more accurate result than 16 (trivially). The third and fourth cases are more interesting because they use mix precisions. When s is float 32 and the value is float16, PyTorch preforms addition in float32 and keeps s in float32., while the error lies in precision of the value (x.type = float16). Ofc casting x to float32 after it assigned to float16 do not recover precision