import itertools
import torch
import argparse
from cs336_basics.model import (BasicsTransformerLM)
from cs336_basics.optimizer import (AdamW)
from cs336_basics.nn_utils import (cross_entropy,softmax)
import timeit
import torch.cuda.nvtx as nvtx
import math
from einops import einsum

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q,K,V,mask=None):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores:"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def main():
    #hyperparameters
    batch_size = 8

    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_prof", action="store_true")
    parser.add_argument("--run",type=str)

    args = parser.parse_args()
    n_steps = 100

    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]
    dimensions = itertools.product(d_models,seq_lengths)

    for d_model,seq_len in dimensions:
        shape = (batch_size,seq_len,d_model)
        try:
            if args.memory_prof:
                torch.cuda.memory._record_memory_history(max_entries=1_000_000,
                                                        enabled="all" if args.memory_prof else None)
            Q = K = V = output = loss = None
            Q = torch.randn(shape,device="cuda",dtype=torch.float32,requires_grad=True)
            K = torch.randn(shape,device="cuda",dtype=torch.float32,requires_grad=True)
            V = torch.randn(shape,device="cuda",dtype=torch.float32,requires_grad=True) 
            for _ in range(5):
                output = annotated_scaled_dot_product_attention(Q,K,V)
                loss = output.sum()
                loss.backward()
                torch.cuda.synchronize()
                output = loss = None
            Q.grad = K.grad = V.grad = None
            #forward       
            start_time = timeit.default_timer()
            for _ in range(n_steps):
                output = annotated_scaled_dot_product_attention(Q,K,V)
                torch.cuda.synchronize()
                output = None
            elapsed_time = timeit.default_timer() - start_time
            if args.memory_prof:
                snapshot_name = f"artifacts/attention_bechmark/f_memory_{args.run}_d{d_model}_s{seq_len}.pickle"
                torch.cuda.memory._dump_snapshot(snapshot_name)
                torch.cuda.memory._record_memory_history(enabled=None)
                
    
            #backward
            torch.cuda.memory._record_memory_history(max_entries=1_000_000,
                                                enabled="all" if args.memory_prof else None)
            backward_time = 0.0
            for _ in range(n_steps):
                Q.grad = K.grad = V.grad = None
                output = annotated_scaled_dot_product_attention(Q,K,V)
                loss = output.sum()
                torch.cuda.synchronize()
    
                memory_before_backward = torch.cuda.memory_allocated()
    
                start_time = timeit.default_timer()
                loss.backward()
                torch.cuda.synchronize()
                backward_time += timeit.default_timer() - start_time
    
                output = loss = None
            if args.memory_prof:
                snapshot_name = f"artifacts/attention_bechmark/b_memory_{args.run}_d{d_model}_s{seq_len}.pickle"
                torch.cuda.memory._dump_snapshot(snapshot_name)
                torch.cuda.memory._record_memory_history(enabled=None)
            print(
                f"d={d_model}, seq={seq_len}: "
                f"forward={elapsed_time/n_steps *1000:.3f} ms, "
                f"backward={backward_time/n_steps*1000:.3f} ms, "
                f"memory={memory_before_backward/1024**2:.2f} MiB"
            )

        except torch.cuda.OutOfMemoryError:
            if args.memory_prof:
                snapshot_name = f"artifacts/attention_bechmark/OOM_b_memory_{args.run}_d{d_model}_s{seq_len}.pickle"
                torch.cuda.memory._dump_snapshot(snapshot_name)
                torch.cuda.memory._record_memory_history(enabled=None)
            print(f"d={d_model}, seq={seq_len}: OOM")
            
            if Q is not None:
                del Q
            if K is not None:
                del K
            if V is not None:
                del V
            if output is not None:
                output = None
            if loss is not None:
                loss = None
                
if __name__ == "__main__":
    main()