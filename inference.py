import os
import torch

from transformers import GPT2TokenizerFast

from model import GPT2Model
from checkpoint import GPTContext

import sys
sys.stdout.reconfigure(encoding='utf-8') # for pycharm output

path = "checkpoint/"
file_name = "1513_time_1775967355.pt"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",local_files_only=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_for_infer(model_class):
    pt_path = os.path.join(path, file_name)
    if not os.path.exists(pt_path):
        raise FileNotFoundError("not found model file:" + pt_path)

    checkpoint = torch.load(pt_path, map_location=device)
    ctx_dict = checkpoint['ctx_dict']
    ctx = GPTContext.from_dict(ctx_dict)
    model = model_class(ctx)
    model.load_state_dict(checkpoint['model_state'])

    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']

    print(f"load checkpoint from {pt_path}, \ntrain loss:{train_loss}, \n"
          f"epoch:{epoch},\n{sum(p.numel() for p in model.parameters())/1e6}M parameters")
    return model

model = load_model_for_infer(GPT2Model)
model.eval()

input_str = "I'm a large language model,"
token_ids = tokenizer.encode(input_str)
print(tokenizer.decode(token_ids), end='')
token_ids = torch.tensor(token_ids).reshape(1, -1)

max_length = 512

while len(token_ids[0]) < max_length:
    new_token_id = model.next_token(token_ids)
    print(tokenizer.decode(new_token_id), end='')
    new_token_id = torch.tensor([[new_token_id]])
    token_ids = torch.cat((token_ids, new_token_id), dim=-1)