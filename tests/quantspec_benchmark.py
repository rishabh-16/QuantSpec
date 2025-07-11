import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from QuantSpec_magidec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from QuantSpec_magidec.Data.data_converter import convert_pg19_dataset, convert_multilexsum_dataset, load_infbench
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
import contextlib
from QuantSpec_magidec.Engine.backend_quantspec import LMBackend
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import marlin
# torch.compiler.allow_in_graph(marlin.marlin_cuda.mul)
# torch._dynamo.symbolic(marlin.marlin_cuda.mul)


parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-hf/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='model name')
parser.add_argument('--marlin_path', type=Path, default=None, help='marlin path')
parser.add_argument('--dataset', type=str, default="pg19", help='dataset')
parser.add_argument('--wq', action='store_true', help='Whether to use weight quantization.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=5, help='start')

parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=4000, help='Prefix length')
parser.add_argument('--gen_len', type=int, default=64, help='Generate length')

parser.add_argument('--seed', type=int, default=125, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')

# Assert max length <= max context length
args = parser.parse_args()
# assert args.prefix_len + args.gen_len + args.gamma + 1 <= 4096

# Init model parallelism
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from QuantSpec_magidec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
if use_tp:
    rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        print = lambda *args, **kwargs: None
else:
    rank = 0

setup_seed(args.seed)

if args.benchmark:
    print(f"Using device={DEVICE}")
MAX_LEN_TARGET = args.prefix_len + args.gen_len + args.gamma
DTYPE = torch.half
BATCH_SIZE = args.B
benchmark = args.benchmark
checkpoint_path = args.model

target_dec_list = [args.gamma + 1]
draft_dec_list = [1]

# Load target model
engine = LMBackend(dtype=DTYPE, device=DEVICE, dec_list=target_dec_list, draft_dec_list=draft_dec_list)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group, quantize=args.wq, marlin_checkpoint=args.marlin_path)
vocab_size = engine.model.config.vocab_size
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET)

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
if args.benchmark:
    print(f"eot_1: {eot_1}, eot_2: {eot_2}")
repeats = 20
no_runs = int(BATCH_SIZE*repeats)
if args.dataset == "pg19":
    dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len) #, end=no_runs)
elif args.dataset == "multilexsum":
    dataset = convert_multilexsum_dataset(tokenizer=tokenizer, seq_len=args.prefix_len) #, end=no_runs)
elif args.dataset == "infbench":
    dataset = load_infbench(tokenizer=tokenizer, seq_len=args.prefix_len) #, end=no_runs)
else:
    raise ValueError(f"Unknown dataset {args.dataset}")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(10, len(dataloader))

total_time = 0.0
num_gen_tokens = 0
target_steps = 0
if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0

# Initialize counters for accepted tokens and total tokens
acceptance_rates = []

for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    # torch.cuda.reset_peak_memory_stats()  # Reset before tracking
    accepted_tokens_count = 0
    total_tokens_count = 0
    if step >= num_eval_steps:
        break
    input_ids = batch[0].to(DEVICE)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
    output = torch.zeros(BATCH_SIZE, args.prefix_len + args.gen_len + args.gamma + 1, device=DEVICE).long()
    output[:, :input_ids.shape[1]] = input_ids
    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]

    tokens_buffer[:,:1] = engine.encode(input_ids=input_ids)[:,-1]
    
    # tokens_buffer[:,:1] = sampling_argmax_batch(logits=logits)
    
    cachelens_update = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    while terminal == False:

        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        for i in range(args.gamma):
            tokens_buffer[:,i+1:i+2] = engine.draft_inference(tokens_buffer[:, i].view(-1,1))
            # tokens_buffer[:,i+1:i+2] = draft_sample[1](engine.draft_inference(tokens_buffer[:, i].view(-1,1)))

        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            draft_time+=t2-t1

        engine.cachelens = engine.cachelens - args.gamma

        # Target Verification    
        target_tokens = engine.inference(tokens_buffer)    


        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            target_time+=t3-t2

        # target_tokens = target_sample(target_logits)
        target_steps+=1

    # Verify loop
        bonus_tokens = torch.full((BATCH_SIZE, 1), 0, device=DEVICE).long()
        accept_nums = torch.full((BATCH_SIZE, 1), 1, device=DEVICE).long()
        accept_flags = torch.full((BATCH_SIZE, 1), True, device=DEVICE)

        draft_tokens = tokens_buffer[:, 1:args.gamma+1]
        flag_accept_matrix = (target_tokens[:, :args.gamma] == draft_tokens)  # shape: (BATCH_SIZE, gamma)
        eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))  # shape: (BATCH_SIZE, gamma)
        accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
        accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
        accept_flags_matrix = accept_flags_cumprod.bool()
        accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True) + 1  # shape: (BATCH_SIZE, 1)
        accepted_tokens_count += accept_flags_matrix.sum()
        total_tokens_count += args.gamma * BATCH_SIZE
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any():
            terminal = True
                
        # Rollback the memory length
        engine.cachelens = engine.cachelens - args.gamma - 1

        # Put the accepted tokens to output
        positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        output[mask] = tokens_buffer[mask_buffer]

        # Set the cache length to the accepted length
        engine.cachelens += accept_nums.flatten()
        
        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == 2).any() or (bonus_tokens == 0).any():
            terminal = True
        num_nodes += accept_nums.flatten()

        # Check Number of Nodes + Bonus Token <= max_target_token
        if num_nodes.max() + 1 >= args.prefix_len + args.gen_len:
            terminal = True
        # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
        if not terminal:
            tokens_buffer[:, :1] = bonus_tokens
        
        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3
        else:
            for i in range(BATCH_SIZE):
                output[i, num_nodes[i]] = bonus_tokens[i]
            num_nodes += 1
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3

    torch.cuda.synchronize()
    end=time.perf_counter()
    total_time += end-start
    num_gen_tokens += (num_nodes.sum() - (input_ids.shape[1]+1)*BATCH_SIZE)
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i, args.prefix_len:num_nodes[i]]))
    if benchmark:
        print("acceptance rate: {:.2%}, total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}".format(accepted_tokens_count / total_tokens_count, total_time, total_time / target_steps, num_gen_tokens, target_steps))
    acceptance_rates.append(accepted_tokens_count / total_tokens_count)
    if benchmark:
        print("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {}".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))
        print(f"Tokens per second :{BATCH_SIZE*(num_gen_tokens/total_time)}")
    if step < 3:   # TODO: revert to 10?
        total_time = 0.0
        num_gen_tokens = 0
        target_steps = 0
        acceptance_rates = []
        if benchmark:
            draft_time = 0.0
            target_time = 0.0
            verify_loop = 0.0
    if use_tp:
        dist.barrier()
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to MB
    # print(f"Peak GPU Memory Usage: {peak_memory:.2f} GB")

# Calculate acceptance rate
acceptance_rate = sum(acceptance_rates) / len(acceptance_rates) if len(acceptance_rates) > 0 else 0
print(f"Acceptance Rate: {acceptance_rate:.2%}")
# print(f"method latency: {total_time/num_gen_tokens}")
print(f"Tokens per second :{BATCH_SIZE*(num_gen_tokens/total_time)}")