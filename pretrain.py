import os.path

import torch
import torch.nn.functional as F
import time

from dataloader import SimpleDataloader
from model import GPT2Model
from other import GPTContext
from tokenizer import SimpleTokenizer
import math

torch.manual_seed(1337)
file_name = ""
file_path = ""
dict_path = ""

eval_iters = 5000
eval_interval = 1

batch_size = 4
block_size = 1024
vocab_size = 50257
grad_clip = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

simple_data_loader = SimpleDataloader(dict_path, file_path, batch_size,block_size, device)

simple_tk = SimpleTokenizer()
simple_tk.load_vocab(dict_path)

simple_data_loader.initialize()
print(f"vocab size:{simple_tk.vocab_size}, total token:{simple_data_loader.total_token_num}")

start_time = time.time()

ctx = GPTContext()

cur_path = os.getcwd()
check_point_path = cur_path + "/{}_{}_{}" # file_name,iter,timestamp
need_checkpoint = True

ctx.block_size = block_size
ctx.vocab_size = vocab_size
ctx.n_layer = 12
ctx.embedding_dim = 768
ctx.head_num = 12

model = GPT2Model(ctx)
model.to(device)
param_num = sum(p.numel() for p in model.parameters())
print(f"{param_num / 1e6}M parameters, {param_num}")
print(f"device:{device}")

## 1 dynamic learn rate
## 2 beta1 beta2
## 3 weight decay
## 4 warm up
max_lr=6e-4
# 5000 and 0.3 is just for debug, need rethinking later
total_steps = 5000
warmup_steps = total_steps * 0.3


def get_lr(step, warmup_steps, total_steps, lr_max):
    if step < warmup_steps:
        return lr_max * step / warmup_steps

    step = min(step, total_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr_min = 0.1 * lr_max

    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

optimizer = torch.optim.AdamW(model.parameters(),
    lr=max_lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1) #todo: only sett weight decay for two dimension params

# mode="reduce-overhead" ? fullgraph?
# model = torch.compile(model,fullgraph=True)

# global_batch_size = 524288
global_batch_size = 65536
micro_batch_size = batch_size * block_size
grad_accu_num = global_batch_size // (micro_batch_size)
print(f"global batch size:{global_batch_size}, micro batch size:{micro_batch_size}, grad accu num:{grad_accu_num}")

for i in range(total_steps):
  t0 = time.time()
  lr = get_lr(i + 1, warmup_steps, total_steps, max_lr)

  for param_group in optimizer.param_groups:
      param_group["lr"] = lr

  optimizer.zero_grad(set_to_none=True)

  acc_loss = 0
  token_num = 0
  for j in range(grad_accu_num):
      train_data = simple_data_loader.next_batch()
      xb = train_data[0].to(device)
      yb = train_data[1].to(device)
      token_num += xb.numel()

      with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
          output = model(xb)
          loss = F.cross_entropy(output.view(-1, vocab_size),  yb.view(-1))
      loss = loss / grad_accu_num
      acc_loss += loss.detach().item()
      loss.backward()

  if grad_clip != 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
  optimizer.step()

  allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
  reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
  torch.cuda.reset_peak_memory_stats()

  torch.cuda.synchronize()
  t4 = time.time()
  tokens_per_second = token_num / (t4 - t0)
  print(f"idx:{i}, train loss {acc_loss}, time {(t4 - t0)*1000:.2f}ms, allocated:{allocated:.2f}"
            f",reserved:{reserved:.2f},"
            f"tokens per sec:{tokens_per_second:.2f}")