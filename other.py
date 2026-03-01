import torch
import torch.nn.functional as F
from dataclasses import dataclass

import os

@dataclass
class GPTContext:
  num_hidden_layer:int=2
  block_size:int=10
  embedding_dim:int=32
  head_num:int=4
  head_dim:int=8
  vocab_size:int = 0
  dropout:float=0
  bias:bool =True


@staticmethod
def to_dict(ctx):
      config_dict = {
          'num_hidden_layer':ctx.num_hidden_layer,
          'block_size': ctx.block_size,
          'embedding_size':ctx.embedding_size,
          'head_num':ctx.head_num,
          'head_dim':ctx.head_dim,
          'mlp_hidden_size':ctx.mlp_hidden_size,
          'vocab_size':ctx.vocab_size,
          'dropout':ctx.dropout
      }
      return config_dict

@staticmethod
def from_dict(config_dict):
      ctx = GPTContext()
      ctx.block_size = config_dict['block_size']
      ctx.embedding_size = config_dict['embedding_size']
      ctx.head_num = config_dict['head_num']
      ctx.head_dim = config_dict['head_dim']
      ctx.mlp_hidden_size = config_dict['mlp_hidden_size']
      ctx.num_hidden_layer = config_dict['num_hidden_layer']
      ctx.vocab_size = config_dict['vocab_size']
      ctx.dropout = config_dict['dropout']
      return ctx

def get_batch(split, train_data,val_data,block_size,batch_size,device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def save_checkpoint(cp_path, model, optimizer, epoch, train_loss, eval_loss
                    ,config_dict):
    os.makedirs(os.path.dirname(cp_path), exist_ok=True)
    check_point = {
        'epoch':epoch,
        'model_state':model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_loss':eval_loss,
        'ctx_dict':config_dict
    }

    torch.save(check_point, cp_path)
    print(f"save checkpoint in {cp_path}")

def load_checkpoint(cp_path, model, optimizer, device):
    if not os.path.exists(cp_path):
        raise FileNotFoundError(f"checkpoint file not found: {cp_path}")

    checkpoint = torch.load(cp_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    epoch = checkpoint['epoch']
    eval_loss = checkpoint['eval_loss']
    train_loss = checkpoint['train_loss']

    print(f"load checkpoint from {cp_path}, train loss:{train_loss}, eval loss:{eval_loss}")
    return epoch


@torch.no_grad()
def estimate_loss(model, eval_iters, vocab_size,train_data,val_data,block_size,batch_size,device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split,train_data,val_data,block_size,batch_size,device)
            logits = model(X)

            logits = logits.view(-1, vocab_size)
            labels = Y.view(-1)
            loss = F.cross_entropy(logits, labels)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out