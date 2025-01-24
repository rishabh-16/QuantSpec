import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from QuantSpec_magidec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from QuantSpec_magidec.Data.data_converter import convert_pg19_dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
import contextlib
from QuantSpec_magidec.Engine.backend_quantspec import LMBackend
from QuantSpec_magidec.Engine.utils import spec_stream

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-hf/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='model name')
parser.add_argument('--marlin_path', type=Path, default=None, help='marlin path')
parser.add_argument('--wq', action='store_true', help='Whether to use weight quantization.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=5, help='start')

parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=4000, help='Prefix length')
parser.add_argument('--gen_len', type=int, default=64, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

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
target_sample = cuda_graph_for_sampling_argmax_batch(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE, idx_len=args.gamma+1, dim=vocab_size)
draft_sample = {}
draft_sample[1] = cuda_graph_for_sampling_argmax_batch(device=DEVICE, dtype=DTYPE, batch_size=BATCH_SIZE, idx_len=1, dim=vocab_size)

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")
repeats = 20
no_runs = int(BATCH_SIZE*repeats)
dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len) #, end=no_runs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(20, len(dataloader))

total_time = 0.0
num_gen_tokens = 0
target_steps = 0
if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0

prof = contextlib.nullcontext()

# Initialize counters for accepted tokens and total tokens
acceptance_rates = []

for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
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

    logits = engine.encode(input_ids=input_ids)[:,-1]
    
    tokens_buffer[:,:1] = sampling_argmax_batch(logits=logits)
    
    if args.printoutput:
        spec_stream(input_ids[0, -50:], tokenizer, 'yellow')
        spec_stream(tokens_buffer[0], tokenizer, 'cyan')
            
    cachelens_update = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    while terminal == False:

        # Draft speculation
        if (step == num_eval_steps - 1) and (rank == 0):
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()

        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        with prof:    
            for i in range(args.gamma):
                tokens_buffer[:,i+1:i+2] = draft_sample[1](engine.draft_inference(tokens_buffer[:, i].view(-1,1)))

        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            draft_time+=t2-t1

        engine.cachelens = engine.cachelens - args.gamma

        # Target Verification    
        target_logits = engine.inference(tokens_buffer)    

        # import IPython
        # IPython.embed()

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            target_time+=t3-t2

        target_tokens = target_sample(target_logits)
        target_steps+=1

        # Verify loop
        bonus_tokens = torch.full((BATCH_SIZE, 1), 0, device=DEVICE).long()
        accept_nums = torch.full((BATCH_SIZE, 1), 1, device=DEVICE).long()
        accept_flags = torch.full((BATCH_SIZE, 1), True, device=DEVICE)
        for pos in range(args.gamma):
            target_token = target_tokens[:, pos]
            draft_token = tokens_buffer[:, pos+1]
            flag_accept = (target_token == draft_token).unsqueeze(1)
            # Ensure flags remain False once they have been set to False
            accept_flags = accept_flags & flag_accept
            # Only increase accept_nums where accept_flags are still True
            accept_nums += accept_flags.int()
            # Count accepted tokens
            accepted_tokens_count += flag_accept.sum().item()
            # Count total tokens processed
            total_tokens_count += BATCH_SIZE

            # Whether or not terminate
            condition = ((draft_token.unsqueeze(1) == eot_1) | (draft_token.unsqueeze(1) == eot_2)) & accept_flags
            if condition.any():
                terminal = True
            accept_flags = accept_flags & ~condition
        
        # Rollback the memory length
        engine.cachelens = engine.cachelens - args.gamma - 1

        # Put the accepted tokens to output
        positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        output[mask] = tokens_buffer[mask_buffer]
        
        if args.printoutput:
            spec_stream(tokens_buffer[mask_buffer][1:], tokenizer, 'green')
            
        # Set the cache length to the accepted length
        engine.cachelens += accept_nums.flatten()
        # max_limit = torch.full_like(accept_nums, args.gamma, device = DEVICE)
        # limited_accept_nums = torch.min(accept_nums, max_limit)
        # engine.draft_cachelens = engine.draft_cachelens - args.gamma
        # # engine.draft_cachelens += accept_nums.flatten()
        # engine.draft_cachelens += limited_accept_nums.flatten()

        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == 2).any() or (bonus_tokens == 0).any():
            terminal = True
        num_nodes += accept_nums.flatten()
        
        if args.printoutput:
            if indices == args.gamma:
                spec_stream(bonus_tokens[0], tokenizer, 'cyan')
            else:
                spec_stream(bonus_tokens[0], tokenizer, 'red')
            
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
    step_time = end-start
    total_time += step_time
    step_gen_tokens = (num_nodes.sum() - (input_ids.shape[1]+1)*BATCH_SIZE)
    num_gen_tokens += step_gen_tokens
    if args.printoutput:
        print("\n" * 2)
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i, args.prefix_len:num_nodes[i]]))
    if benchmark:
        print("acceptance rate: {:.2%}, total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}".format(accepted_tokens_count / total_tokens_count, total_time, total_time / target_steps, num_gen_tokens, target_steps))
    acceptance_rates.append(accepted_tokens_count / total_tokens_count)
    if benchmark:
        print("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {}, Tokens per second: {}".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE, step_gen_tokens / step_time))
    if step < 2:   # TODO: revert to 10?
        total_time = 0.0
        num_gen_tokens = 0
        target_steps = 0
        if benchmark:
            draft_time = 0.0
            target_time = 0.0
            verify_loop = 0.0
    if use_tp:
        dist.barrier()
    
    if step == (num_eval_steps - 2):
        print("We do not benchmark the last sample")
        break

# Calculate acceptance rate
acceptance_rate = sum(acceptance_rates) / len(acceptance_rates) if len(acceptance_rates) > 0 else 0
print(f"Acceptance Rate: {acceptance_rate:.2%}")
print(f"Method latency: {total_time / num_gen_tokens}")
print(f"Method Tokens per second: {num_gen_tokens / total_time}")

if hasattr(prof, "export_chrome_trace"):
    prof.export_chrome_trace(f"prof_selfspec.json")