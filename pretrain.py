import os.path

import torch
import torch.nn.functional as F
import time
import numpy as np
from transformers import GPT2TokenizerFast
import math

from dataloader import SimpleDataloader
from model import GPT2Model
from checkpoint import GPTContext

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = ""
tk_arr = np.memmap(data_path, dtype=np.uint16, mode='r')
total_tk_num = len(tk_arr)

batch_size = 4
block_size = 1024

eval_tokens = 1024 * 1024
eval_batch_size = 4
eval_interval_step = 5 # 300
eval_iters = (eval_tokens // (eval_batch_size * block_size)) - 1

train_data_start_idx = 0
train_data_end_idx = total_tk_num - eval_tokens - 20
eval_data_start_idx = train_data_end_idx + 10
eval_data_end_idx = eval_data_start_idx + eval_tokens - 1

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
train_data_loader = SimpleDataloader(tk_arr,train_data_start_idx,train_data_end_idx,batch_size,block_size)
val_data_loader = SimpleDataloader(tk_arr,eval_data_start_idx,eval_data_end_idx,eval_batch_size,block_size)
print("total tokens:{}".format(total_tk_num))
print(f"total val tokens:{eval_tokens}, "
      f"eval_batch:{eval_batch_size * block_size},"
      f"eval iters:{eval_iters},"
      f"eval start idx:{eval_data_start_idx},"
      f"eval end idx:{eval_data_end_idx}")
vocab_size = tokenizer.vocab_size

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
print(f"{param_num / 1e6}M parameters")
print(f"device:{device}")

def get_lr(step, warmup_steps, total_steps, lr_max):
    if step < warmup_steps:
        return lr_max * step / warmup_steps

    if step > total_steps:
        return min_lr;

    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    return min_lr + 0.5 * (lr_max - min_lr) * (1 + math.cos(math.pi * progress))

def configure_optimizer(model, lr, weight_decay, betas=(0.9, 0.95), eps=1e-8):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_group = [
        {"params":decay_params, "weight_decay":weight_decay},
        {"params":no_decay_params, "weight_decay":0}
    ]

    optimizer = torch.optim.AdamW(param_group,lr,betas,eps)
    return optimizer

@torch.no_grad()
def estimate_val_loss(model, data_loader, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        val_data = data_loader.next_batch()
        x = val_data[0].to(device)
        y = val_data[1].to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(x)
            loss = F.cross_entropy(output.view(-1, vocab_size), y.view(-1))
        losses[i] = loss.item()

    model.train()
    data_loader.reset_start_idx()
    return losses.mean().item()

# mode="reduce-overhead" ? fullgraph?
model = torch.compile(model,fullgraph=True)

# set global_batch_size to 524288 is too big for one gpu training
global_batch_size = 65536
total_train_token_num = train_data_loader.total_token_num() // global_batch_size * global_batch_size
print(f"vocab size:{vocab_size}, total train token:{total_train_token_num}")

micro_batch_size = batch_size * block_size
grad_accu_num = global_batch_size // micro_batch_size
print(f"global batch size:{global_batch_size}, micro batch size:{micro_batch_size}, grad accu num:{grad_accu_num}")

grad_clip = 1.0
max_lr=6e-4
min_lr = max_lr * 0.1
weight_decay = 0.1

total_steps = total_train_token_num // global_batch_size
warmup_steps = 200
print(f"total steps:{total_steps}, warmup steps:{warmup_steps}")

optimizer = configure_optimizer(model, min_lr, weight_decay)

stop_flag = False
for i in range(total_steps):
  if stop_flag:
      break

  t0 = time.time()
  lr = get_lr(i + 1, warmup_steps, total_steps, max_lr)

  for param_group in optimizer.param_groups:
      param_group["lr"] = lr
  optimizer.zero_grad(set_to_none=True)

  acc_loss = 0
  token_num = 0
  for j in range(grad_accu_num):
      train_data = train_data_loader.next_batch()
      if train_data is None:
          stop_flag = True
          break
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
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
  optimizer.step()

  allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
  reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
  torch.cuda.reset_peak_memory_stats()

  torch.cuda.synchronize()
  t4 = time.time()
  tokens_per_second = token_num / (t4 - t0)

  print(f"step:{i}/{total_steps}({i / total_steps * 100:.2f}%),"
        f"train loss:{acc_loss},"
        f"lr:{lr:.2e},"
        f"g_norm:{grad_norm.item():.4f},"
        f"tm:{(t4 - t0) * 1000:.2f}ms,"
        f"alloc:{allocated:.2f},"
        f"reserve:{reserved:.2f},"
        f"tps:{tokens_per_second:.2f}")
  if i != 0 and i % eval_interval_step == 0:
      val_loss = estimate_val_loss(model, val_data_loader, eval_iters)
      print(
        f"step:{i}/{total_steps}({i / total_steps * 100:.2f}%),"
        f"train loss:{acc_loss},"
        f"val loss:{val_loss}")
