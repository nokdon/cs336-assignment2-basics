import argparse
import torch
from cs336_basics.model import (BasicsTransformerLM)
from cs336_basics.optimizer import (AdamW)
from cs336_basics.nn_utils import (cross_entropy)
import timeit

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
            logits_train = model_obj(X_train)
            torch.cuda.synchronize()


    if args.run == "f_b":
        for _ in range(args.w):
            optimizer.zero_grad()
            logits_train = model_obj(X_train)
            loss = cross_entropy(logits_train,Y_train)
            loss.backward()
            torch.cuda.synchronize()
        #-----------
        
        start_time = timeit.default_timer()
        for _ in range(args.n):
            optimizer.zero_grad()
            logits_train = model_obj(X_train)
            loss = cross_entropy(logits_train,Y_train)
            loss.backward()
            torch.cuda.synchronize()


    if args.run == "full":
        for _ in range(args.w):
            optimizer.zero_grad()
            logits_train = model_obj(X_train)
            loss = cross_entropy(logits_train,Y_train)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        #-----------
        
        start_time = timeit.default_timer()
        for _ in range(args.n):
            optimizer.zero_grad()
            logits_train = model_obj(X_train)
            loss = cross_entropy(logits_train,Y_train)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
    #---------------------------------------------------------------------
    elapsed_time = timeit.default_timer() - start_time
    print(f"milliseconds per step {elapsed_time/args.n * 1_000}")
if __name__ == "__main__":
    main()