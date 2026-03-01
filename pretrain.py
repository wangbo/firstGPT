import os.path

import torch
import torch.nn.functional as F
import time
from collections import deque
import sys

from model import GPT2Model, GPTContext, get_batch, estimate_loss, save_checkpoint
from tokenizer import SimpleTokenizer

file_name = "all"
file_path = "/home/wangbo/git/cn_data/" + file_name + ".txt"
dict_path = "/home/wangbo/git/cn_data/" + file_name + ".dict"

simple_tk = SimpleTokenizer()
simple_tk.load_token_dict(dict_path)
train_data, val_data = simple_tk.load_data(file_path)
print(f"vocab size:{simple_tk.vocab_size}, total token:{simple_tk.total_token_num}")

start_time = time.time()
# learning_rate = 3e-4
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 256
batch_size = 32
vocab_size = simple_tk.vocab_size
ctx = GPTContext()
ctx.block_size = block_size
ctx.vocab_size = simple_tk.vocab_size
cur_path = os.getcwd()
check_point_path = cur_path + "/{}_{}_{}" # file_name,iter,timestamp
need_checkpoint = True

eval_iters = 100
train_iters = 10000000000
eval_interval = 500

# model param
ctx.num_hidden_layer = 4
ctx.embedding_size = 128
ctx.head_num = 4
ctx.head_dim = 32
ctx.mlp_hidden_size = 256
# debug param
# ctx.num_hidden_layer = 2
# ctx.embedding_size = 256
# ctx.head_num = 4
# ctx.head_dim = 64
# ctx.mlp_hidden_size = 256

model = GPT2Model(ctx)
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print(f"device:{device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

iter_num = 0

val_check_queue_size = 5
val_loss_check_queue = deque()

last_val_loss = sys.maxsize
for i in range(train_iters):
  xb, yb = get_batch('train',train_data,val_data,block_size,batch_size,device)

  output = model(xb)
  output = output.view(-1, vocab_size)
  labels = yb.view(-1)
  loss = F.cross_entropy(output, labels)
  if i % eval_interval == 0 or i == train_iters - 1:
    est_loss = estimate_loss(model, eval_iters, vocab_size,train_data,val_data,block_size,batch_size,device)
    print(f"idx:{i}, train loss:{est_loss['train']}, eval loss:{est_loss['val']}")
    # check stop condition
    cur_val_loss = round(est_loss['val'].item(), 2)
    stop_flag = cur_val_loss >= last_val_loss
    val_loss_check_queue.append(stop_flag)
    last_val_loss = cur_val_loss
    if len(val_loss_check_queue) > val_check_queue_size:
        val_loss_check_queue.popleft()
    if all(val_loss_check_queue):
        print("meets stop condition, stop training")
        break

  if i != 0 and i % 5000 == 0:
      final_loss = estimate_loss(model, eval_iters, vocab_size, train_data, val_data, block_size, batch_size, device)
      save_checkpoint(check_point_path.format(file_name, str(iter_num), str(int(start_time))),
                          model, optimizer, iter_num, final_loss['train'], final_loss['val'],
                          GPTContext.to_dict(ctx))

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  iter_num = i

final_loss = estimate_loss(model, eval_iters, vocab_size,train_data,val_data,block_size,batch_size,device)

if need_checkpoint:
    save_checkpoint(check_point_path.format(file_name, str(int(start_time)), str(iter_num)),
                    model, optimizer, iter_num, final_loss['train'], final_loss['val'],
                    GPTContext.to_dict(ctx))

input_str = "行者"
model.eval()
print("begin eval")
token_ids = simple_tk.encode(input_str)
output_ids = model.generate(token_ids,block_size)
print(simple_tk.decode(output_ids))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码执行耗时：{elapsed_time:.4f} 秒")