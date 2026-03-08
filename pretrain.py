import os.path

import torch
import torch.nn.functional as F
import time
import sys

from model import GPT2Model
from other import GPTContext, get_batch, estimate_loss, save_checkpoint
from tokenizer import SimpleTokenizer

torch.manual_seed(1337)
file_name = "all"
file_path = "" + file_name + ".txt"
dict_path = "" + file_name + ".dict"

simple_tk = SimpleTokenizer()
simple_tk.load_token_dict(dict_path)
train_data, val_data = simple_tk.load_data(file_path)
print(f"vocab size:{simple_tk.vocab_size}, total token:{simple_tk.total_token_num}")

start_time = time.time()
# learning_rate = 3e-4
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ctx = GPTContext()

cur_path = os.getcwd()
check_point_path = cur_path + "/{}_{}_{}" # file_name,iter,timestamp
need_checkpoint = True


train_iters = 200
eval_iters = 200
eval_interval = 1000

batch_size = 16
block_size = 1024
vocab_size = simple_tk.vocab_size
# vocab_size = 50257

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
# for name,param in model.named_parameters():
#     print(f"{name:20s}, dtype:{param.dtype}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

iter_num = 0

last_val_loss = sys.maxsize
for i in range(train_iters):
  xb, yb = get_batch('train',train_data,val_data,block_size,batch_size,device)

  t0 = time.time()
  with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      output = model(xb)
      loss = F.cross_entropy(output.view(-1, vocab_size),  yb.view(-1))

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  iter_num = i
  t1 = time.time()

  allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
  reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
  print(f"idx:{i}, train loss {loss.item()}, time {(t1 - t0)*1000:.2f}ms, allocated:{allocated}"
        f",reserved:{reserved}")

  if (i != 0 and i % eval_interval == 0) or i == train_iters - 1:
    est_loss = estimate_loss(model, eval_iters, vocab_size,train_data,val_data,block_size,batch_size,device)
    print(f"idx:{i}, train loss:{est_loss['train']}, eval loss:{est_loss['val']}")

  # if i != 0 and i % 2500 == 0:
  #     final_loss = estimate_loss(model, eval_iters, vocab_size, train_data, val_data, block_size, batch_size, device)
  #     save_checkpoint(check_point_path.format(file_name, str(iter_num), str(int(start_time))),
  #                         model, optimizer, iter_num, final_loss['train'], final_loss['val'],
  #                         GPTContext.to_dict(ctx))

final_loss = estimate_loss(model, eval_iters, vocab_size,train_data,val_data,block_size,batch_size,device)

if need_checkpoint:
    save_checkpoint(check_point_path.format(file_name, str(int(start_time)), str(iter_num)),
                    model, optimizer, iter_num, final_loss['train'], final_loss['val'],
                    GPTContext.to_dict(ctx))

input_str = ""
model.eval()
print("begin eval")
token_ids = simple_tk.encode(input_str)
output_ids = model.generate(token_ids,block_size)
print(simple_tk.decode(output_ids))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"time cost：{elapsed_time:.4f} second")