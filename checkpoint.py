
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import time
import os

@dataclass
class GPTContext:
    n_layer:int=2
    block_size:int=10
    embedding_dim:int=32
    head_num:int=4
    vocab_size:int=0
    dropout:float=0
    bias:bool=True
    enable_flash:bool=True

    @staticmethod
    def to_dict(ctx):
        config_dict = {
            'n_layer':ctx.n_layer,
            'block_size': ctx.block_size,
            'embedding_dim':ctx.embedding_dim,
            'head_num':ctx.head_num,
            'vocab_size':ctx.vocab_size,
            'dropout':ctx.dropout,
            'bias':ctx.bias
        }
        return config_dict

    @staticmethod
    def from_dict(config_dict):
        ctx = GPTContext()
        ctx.n_layer = config_dict['n_layer']
        ctx.block_size = config_dict['block_size']
        ctx.embedding_dim = config_dict['embedding_dim']
        ctx.head_num = config_dict['head_num']
        ctx.vocab_size = config_dict['vocab_size']
        ctx.dropout = config_dict['dropout']
        ctx.bias = config_dict['bias']
        return ctx


def save_checkpoint(step, ckpt_path, raw_model, optimizer, train_loss,config_dict):
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Directory does not exist: {ckpt_path}")

    file_name = "{}_time_{}.pt".format(step, int(time.time()))
    full_path = os.path.join(ckpt_path, file_name)

    if os.path.exists(full_path):
        raise FileExistsError(f"File already exists: {full_path}")

    check_point = {
        'epoch':step,
        'model_state':raw_model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'train_loss': train_loss,
        'ctx_dict':config_dict
    }

    torch.save(check_point, full_path)
    print(f"save checkpoint in {full_path}")

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