import os
import torch
from pathlib import Path

from model import GPT2Model, GPTContext
from tokenizer import SimpleTokenizer

file_name = "all"
dict_path = "/home/wangbo/git/cn_data/" + file_name + ".dict"

cp_name = "all_34999_1771763513"
cp_path = "/home/wangbo/git/firstGPT/" + cp_name

simple_tk = SimpleTokenizer()
simple_tk.load_token_dict(dict_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_for_infer(model_class):
    if not Path(cp_path).exists():
        raise FileNotFoundError("not found model file:" + cp_path)

    checkpoint = torch.load(cp_path, map_location=device)
    ctx_dict = checkpoint['ctx_dict']
    ctx = GPTContext.from_dict(ctx_dict)
    model = model_class(ctx)
    model.load_state_dict(checkpoint['model_state'])

    epoch = checkpoint['epoch']
    eval_loss = checkpoint['eval_loss']
    train_loss = checkpoint['train_loss']

    print(f"load checkpoint from {cp_path}, \ntrain loss:{train_loss}, eval loss:{eval_loss},\n"
          f"epoch:{epoch},\n{sum(p.numel() for p in model.parameters())/1e6}M parameters")
    return model

model = load_model_for_infer(GPT2Model)
model.eval()

input_str = "孙行者"
token_ids = simple_tk.encode(input_str)
output_ids = model.generate(token_ids, 30)
print(simple_tk.decode(output_ids))