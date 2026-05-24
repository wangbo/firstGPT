from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from inference import load_model_for_infer
from model import GPT2Model

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

torch.manual_seed(1337)

model_name = "gpt2"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
logging.info(f"vocab size:{tokenizer.vocab_size}")
# gpt2_raw_model = AutoModelForCausalLM.from_pretrained(
#     model_name,local_files_only=True).to(device)

eos_token_id = tokenizer.eos_token_id

# ppl is exp for forward loss
@torch.no_grad()
def eval_ppl(token_list, model):
    block_size = 1024
    stride = 512
    model.eval()
    losses = torch.tensor([], dtype=torch.float32)
    losses = losses.to(device)
    
    for i in range(0, len(token_list), stride):
        batch = token_list[i : i + block_size]
        if len(batch) < block_size:
            break
        
        input_ids = torch.tensor(batch, dtype=torch.long).unsqueeze(0)
        input_ids = input_ids.to(device)
        labels = torch.tensor(token_list[i + 1 : i + 1 + block_size], 
                              dtype=torch.long).unsqueeze(0)
        labels = labels.to(device)

        if i == 0:
            logging.info("first batch")
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(labels.shape[-1]),
            reduction="none"
        )

        if i == 0:
            losses = torch.cat([losses, loss], dim=0)
        else:
            losses = torch.cat([losses, loss[stride:block_size]], dim=0)
        
    
    logging.info(f"losses count:{len(losses)}")
    ppl = torch.exp(torch.mean(losses))
    return ppl

def get_wikitext_token_list():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    logging.info(dataset)

    token_list = []
    # test/train/validation
    for text in dataset['train']['text']:
        tokens = tokenizer.encode(text)
        if len(tokens) == 0:
            continue
        token_list.extend(tokens)
        token_list.append(eos_token_id)
    logging.info(f"wiki text token size:{len(token_list)}")
    return token_list

# ppl = eval_ppl(token_list, gpt2_raw_model)
# logging.info(ppl)
# raw gpt2的计算结果
# 固定窗口计算出来的值 tensor(46.4863
# 滑动窗口步长为512时计算出来的值 tensor(42.6428

my_gpt2_model = load_model_for_infer(GPT2Model,
    arg_path="checkpoint/0501",
    arg_file_name="18983_time_1777921873.pt").to(device)
# ppl = eval_ppl(get_wikitext_token_list(), my_gpt2_model)
# logging.info(ppl)
# wiki text计算出的困惑度是 tensor(53.5905

# 评估时的精度是不是要和训练时保持一致
@torch.no_grad()
def eval_hella_swag(model):
    model.eval()
    hwg_dataset = load_dataset("Rowan/hellaswag", split="validation")
    correct_sample_num = 0
    for sample in tqdm(hwg_dataset):
        ctx = sample['ctx']
        endings = sample['endings']
        sample_scores = []
        correct_opt = int(sample['label'])
        ctx_ids = tokenizer.encode(ctx,add_special_tokens=False)
        for ending in endings:
            ending_ids = tokenizer.encode(" " + ending,add_special_tokens=False)
            ctx_len = len(ctx_ids)
            input_ids = ctx_ids + ending_ids
            
            labels = input_ids[ctx_len:]

            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            logits = model(input_ids=input_ids).logits
            logits = logits[0][ctx_len - 1:-1]
            probs = torch.log_softmax(logits, dim=-1) # 区别log_softmax/softmax
            probs = probs[torch.arange(len(labels)), labels]
            sample_scores.append(probs.mean().item())
        model_opt = sample_scores.index(max(sample_scores))
        if model_opt == correct_opt:
            correct_sample_num += 1
    
    logging.info(f"correct count:{correct_sample_num},"
                 f"rate: {correct_sample_num/len(hwg_dataset)*100:.2f}%")

# gpt2原生：correct count:2967,rate: 29.55%
# eval_hella_swag(gpt2_raw_model)
eval_hella_swag(my_gpt2_model)
# 自己训练的10B：correct count:2893,rate: 28.81%

def train_healthy_check():
    train_log = "log/10B_train0501.log"

    train_step_arr = []
    train_loss_arr = []
    val_step_arr = []
    val_loss_arr = []
    with open(train_log, "r", encoding="utf-8") as file:
        for line in file:
            if "val loss" in line:
                match = re.search(
                    r"step:(\d+)/\d+.*?val loss:([\d\.eE+-]+)",
                    line
                )
                if match:
                    step = int(match.group(1))
                    # if step < 10000:
                    #     continue
                    val_loss = float(match.group(2))
                    print(f"step:{step},val loss:{val_loss}")
                    val_loss_arr.append(val_loss)
                    val_step_arr.append(step)
            else:
                match = re.search(
                    r"step:(\d+)/\d+.*?train loss:([\d\.eE+-]+)",
                    line
                )

                if match:
                    step = float(match.group(1))
                    if step < 10000:
                        continue
                    train_loss = match.group(2)
                    train_step_arr.append(step)
                    train_loss_arr.append(train_loss)

    train_step_arr = np.array(train_step_arr, dtype=np.int32)
    train_loss_arr = np.array(train_loss_arr, dtype=np.float32)
    val_loss_arr = np.array(val_loss_arr, dtype=np.int32)
    val_step_arr = np.array(val_step_arr, dtype=np.float32)

    plt.figure(figsize=(10, 6))
    plt.plot(train_step_arr, train_loss_arr, label="train")
    plt.plot(val_step_arr, val_loss_arr, marker="o", label="val")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()

    plt.show()

# train_healthy_check()
