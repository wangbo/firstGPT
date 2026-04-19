import os
import torch

from transformers import GPT2TokenizerFast

from model import GPT2Model
from checkpoint import GPTContext
import torch.nn.functional as F

import sys
sys.stdout.reconfigure(encoding='utf-8') # for pycharm output

path = "checkpoint/"
file_name = "38134_time_1776436554.pt"

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
# claude prompt
prompts = [
    # "She is going to the store. He is going to the",    # 测语法规律
    # "a b c d e f g h i j k",    # 测重复模式
    # "The quick brown fox jumps over the lazy",    # 测标点/句式闭合
    # "1, 2, 3, 4, 5, 6, 7, 8, 9,",# 测数字规律
    "It was raining outside, so she decided to bring her",    # 测简单因果
]
# [0.1,0.7,1.0]
temps = [0]
# temps = [0.7]
# temps = [1.0]

# gpt prompt
prompts = [
    # "A dog is",
    # "What is a chair?A chair is",
    # "People wear coats in winter because",
    # "Tom has a ball. He gives it to Anna. Who has the ball?The ball is with",
    # "The farmer is working in the field."
    "Repeat the word 'cat' 5 times:"
]
# temps = [0.2]
temps = [0.2]

# todo: batch inference
batch_token_ids = [tokenizer.encode(prompt) for prompt in prompts]

max_length = 1024

for temp in temps:
    print(f"current temp:{temp}")
    for idx, token_ids in enumerate(batch_token_ids):
        token_ids = torch.tensor([token_ids])
        print(f"idx:{idx}")
        print(prompts[idx], end='')
        while len(token_ids[0]) < max_length:
            logits = model.forward(token_ids)
            if temp == 0:
                new_token_id = torch.argmax(logits[0][-1], keepdim=True)
            else:
                vocab = logits[0][-1] / temp
                vocab = F.softmax(vocab, dim=-1)
                new_token_id = torch.multinomial(vocab, num_samples=1, replacement=False)

            print(tokenizer.decode(new_token_id), end='')
            new_token_id = torch.tensor([[new_token_id]])
            token_ids = torch.cat((token_ids, new_token_id), dim=-1)
        print()

