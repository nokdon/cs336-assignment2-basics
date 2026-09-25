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
Problem (benchmarking_mixed_precision):
    a)
    1) model parameters: fp32
    2) first ffl: fp16
    3) layer_norm: fp32
    4) logits: fp16
    5) loss: fp32 if cross_entropy
    6) gradient: fp32 if we are talking about model.parameters and .grad
    b)
    Still we gonna use fp32 for layernorm. bf16 has much broader range than fp16, but precision is worse. For layer norm precision is essential due to computing mean and variance which are sensetive to rounding
    c)  MODEL       fp32|ms    bf16|ms
        small        57,94       44,04        1,32×
        medium      160,33       81,55        1,97×
        large       369,70      130,50        2,83×
        xl         1044,42      213,65        4,89×
        10B        3660,03      590,66        6,20×
    BF16 was faster for all five models. Icreasing from 1.32x for small to 6.20x for 10B. I think wider layers benefit more because their matmul work grows faster than many elementwise operations, and BF16 makes these matmuls faster. Larger matmuls can also use GPU more efficiently making launching operations relatively less important
Problem (memory_profiling):
    a) context_length 128: inference memory usage is almost constant except few small spikes. Weights occupy most of memory and activations aren't saved for backward. Full training: memory grows slightly during forward, much bigger grow during backward and stays near its peak with small spikes on optimizer step. Before next forward pass memory drops due to zero_grad
    b)  f_ctx128_fp32:        12.8 GB
        f_ctx2048_fp32:       14.9 GB
        full_ctx128_fp_32:    51.3 GB
        full_ctx2048_fp_32:   91.3 GB
        For ctx2048, activations that grow linearly with context are 16x bigger, but attention matrices are 16*16=256x bigger compared to ctx128. So the higher peaks due to storing these activations, not gradients itself. And the big drop during backward comes from freeing forward activations, not from clearing gradients
    c)  f_ctx128_fp32:        12.8 GB | bf16: 19.0 GB | +49.3%
        f_ctx2048_fp32:       14.9 GB | bf16: 20.5 GB | +37.2%
        full_ctx128_fp_32:    51.3 GB | bf16: 51.3 GB | +0%
        full_ctx2048_fp_32:   91.3 GB | bf16: 82.4 GB | -9.5%
        Mixed precision does not always reduce memory usage. Autocast keeps the original FP32 weights and creates BF16 copies for some operations, which adds memory and explains the higher inferencepeaks. However, some activations are stored only in BF16, so at ctx2048 the savings on activations outweigh the extra copies and reduce the training peak, while at ctx128 it stays almost unchanged
    d) batch_size x context_length x d_model  * 4 / 1024^2
        -> ctx128: 1.25 MiB | ctx2048 -> 20 MiB . batch_size = 1
    e) ctx2048: the largest allocation I observed were 512 MiB and each came from tensors used in attention computation: einsumm, masking(torch.where), sotmax