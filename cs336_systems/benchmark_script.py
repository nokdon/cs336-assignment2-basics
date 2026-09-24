import argparse
import torch
from cs336_basics.model import (BasicsTransformerLM)
from cs336_basics.optimizer import (AdamW)
from cs336_basics.nn_utils import (cross_entropy,softmax)
import timeit
import torch.cuda.nvtx as nvtx
import einx
import cs336_basics.model
import math
from einops import einsum, rearrange

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q,K,V,mask):
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
    #Hyperparameters-----------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=str)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--d_ff",type=int,default=3072)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int,default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str,default="cuda:0")
    parser.add_argument("--w", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--autocast_bf16", action="store_true")

    args = parser.parse_args()

    assert args.run == "f" or args.run == "f_b" or args.run == "full"

    #Generate random batch of data----------------------------------

    data_ids = torch.randint(low=0,high=args.vocab_size,
                             size=(args.batch_size,
                                   args.context_length+1),
                            dtype=torch.long,device=args.device)
    X_train = data_ids[:,:-1]
    Y_train = data_ids[:, 1:]

    #Run model-------------------------------------------------------

    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    model_obj = BasicsTransformerLM(args.vocab_size,args.context_length,
                args.d_model,args.num_layers,args.num_heads,
                args.d_ff).to(args.device)

    optimizer = AdamW(model_obj.parameters())

    #Modes-------------------------------------------------------------
    if args.run == "f":
        for _ in range(args.w):
            logits_train = model_obj(X_train)
            torch.cuda.synchronize()

        start_time = timeit.default_timer()
        #-----------

        for _ in range(args.n):
            with nvtx.range("measured_step"):
                with nvtx.range("forward"):
                    logits_train = model_obj(X_train)
                torch.cuda.synchronize()


    if args.run == "f_b":
        for _ in range(args.w):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=args.autocast_bf16):
                logits_train = model_obj(X_train)
                loss = cross_entropy(logits_train,Y_train)
            loss.backward()
            torch.cuda.synchronize()
        #-----------
        start_time = timeit.default_timer()

        for _ in range(args.n):
            with nvtx.range("measured_step"):
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=args.autocast_bf16):
                    with nvtx.range("forward"):
                        logits_train = model_obj(X_train)
                    with nvtx.range("loss"):
                        loss = cross_entropy(logits_train,Y_train)
                with nvtx.range("backward"):
                    loss.backward()
                torch.cuda.synchronize()


    if args.run == "full":
        for _ in range(args.w):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=args.autocast_bf16):
                logits_train = model_obj(X_train)
                loss = cross_entropy(logits_train,Y_train)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        #-----------

        start_time = timeit.default_timer()
        for _ in range(args.n):
            with nvtx.range("measured_step"):
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=args.autocast_bf16):
                    with nvtx.range("forward"):
                        logits_train = model_obj(X_train)
                    with nvtx.range("loss"):
                        loss = cross_entropy(logits_train,Y_train)
                with nvtx.range("backward"):
                    loss.backward()
                with nvtx.range("optimizer_step"):
                    optimizer.step()
                torch.cuda.synchronize()
    #---------------------------------------------------------------------
    elapsed_time = timeit.default_timer() - start_time
    print(f"milliseconds per step {elapsed_time/args.n * 1_000}")
if __name__ == "__main__":
    main()