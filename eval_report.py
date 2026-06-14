from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast
from datasets import load_dataset
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from checkpoint import GPTContext
from dataloader import SimpleDataloader
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

def get_gpt2_raw_model():
    return AutoModelForCausalLM.from_pretrained(
        model_name,local_files_only=True).to(device)

eos_token_id = tokenizer.eos_token_id

# ppl is exp for forward loss
@torch.no_grad()
def eval_ppl(token_list, model):
    block_size = 1024
    # stride = 512
    stride = 1024
    model.eval()
    losses = torch.tensor([], dtype=torch.float32)
    losses = losses.to(device)
    
    for i in range(0, len(token_list), stride):
        batch = token_list[i : i + block_size]
        if len(batch) < block_size:
            break
        
        input_ids = torch.tensor(batch, dtype=torch.long).unsqueeze(0)
        input_ids = input_ids.to(device)

        label_begin_idx = i + 1
        label_end_idx = i + 1 + block_size
        labels = torch.tensor(token_list[label_begin_idx:label_end_idx],
                              dtype=torch.long).unsqueeze(0)
        if label_end_idx >= len(token_list):
            logging.info(f"end_idx={label_end_idx}, len_labels={len(labels)}")
            break
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

        if i == 0 or stride == block_size:
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

def get_val_token_list():
    #begin idx = 9952940756, end idx = 9953989331
    data_path = "../fineweb10t/sample/10B.bin"
    origin_token_arr = np.memmap(data_path, dtype=np.uint16, mode='r')
    val_token_arr = origin_token_arr[9952940756:9953989331 + 1]
    logging.info(f"val token list:{len(val_token_arr)}")
    return val_token_arr

# ppl = eval_ppl(token_list, gpt2_raw_model)
# logging.info(ppl)
# raw gpt2的计算结果
# 固定窗口计算出来的值 tensor(46.4863
# 滑动窗口步长为512时计算出来的值 tensor(42.6428

def get_gpt2_model():
    return load_model_for_infer(GPT2Model,
        arg_path="checkpoint/0501",
        arg_file_name="18983_time_1777921873.pt").to(device)

# my_gpt2_model = get_gpt2_model()
# ppl = eval_ppl(get_wikitext_token_list(), my_gpt2_model)
# logging.info(ppl)
# wiki text计算出的困惑度是 tensor(53.5905

# 评估同分布数据集的困惑度
# tensor(19.9250, 滑动窗口为512
# 滑动窗口为1024时, 困惑度为21.1271, 求log时3.056
# ppl = eval_ppl(get_val_token_list(), my_gpt2_model)
# logging.info(ppl)

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
            log_probs = torch.log_softmax(logits, dim=-1) # 区别log_softmax/softmax
            log_probs = log_probs[torch.arange(len(labels)), labels]
            sample_scores.append(log_probs.mean().item())
        model_opt = sample_scores.index(max(sample_scores))
        if model_opt == correct_opt:
            correct_sample_num += 1
    
    logging.info(f"correct count:{correct_sample_num},"
                 f"rate: {correct_sample_num/len(hwg_dataset)*100:.2f}%")

# gpt2原生：correct count:2967,rate: 29.55%
# eval_hella_swag(gpt2_raw_model)
# eval_hella_swag(my_gpt2_model)
# 自己训练的10B：correct count:2893,rate: 28.81%

def train_healthy_check():
    train_log = "log/10B_train0501.log"

    train_step_arr = []
    train_loss_arr = []
    val_step_arr = []
    val_loss_arr = []
    gnorm_arr = []
    lr_arr = []
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
                    # print(f"step:{step},val loss:{val_loss}")
                    val_loss_arr.append(val_loss)
                    val_step_arr.append(step)
            else:
                pattern = r"step:(\d+)/\d+.*?train loss:([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),lr:([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),g_norm:([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
                match = re.search(pattern,line)

                if match:
                    step = int(match.group(1))
                    train_loss = float(match.group(2))
                    lr = float(match.group(3))
                    g_norm = float(match.group(4))

                    train_step_arr.append(step)
                    train_loss_arr.append(train_loss)
                    gnorm_arr.append(g_norm)
                    lr_arr.append(lr)
                    # print(f"step:{step},train loss:{train_loss}")

    train_step_arr = np.array(train_step_arr, dtype=np.int32)
    train_loss_arr = np.array(train_loss_arr, dtype=np.float32)
    gnorm_arr = np.array(gnorm_arr, dtype=np.float32)
    lr_arr = np.array(lr_arr, dtype=np.float32)
    val_loss_arr = np.array(val_loss_arr, dtype=np.int32)
    val_step_arr = np.array(val_step_arr, dtype=np.float32)

    plt.figure(figsize=(10, 6))
    plt.plot(train_step_arr, train_loss_arr, label="train")
    # plt.plot(val_step_arr, val_loss_arr, marker="o", label="val")
    plt.plot(train_step_arr, gnorm_arr, marker="o", label="gnorm")
    plt.plot(train_step_arr, lr_arr, marker="o", label="lr")

    plt.xlabel("Step")
    plt.ylabel("lr")
    plt.title("lr")
    plt.legend()
    plt.grid()

    plt.show()

train_healthy_check()

def check_val_loss(mode, dropout):
    data_path = "../fineweb10t/sample/10B.bin"
    origin_token_arr = np.memmap(data_path, dtype=np.uint16, mode='r')
    val_data_loader = SimpleDataloader(origin_token_arr, 9952940756, 9953989331, 4,
                                       1024)

    checkpoint = torch.load("checkpoint/0501/18983_time_1777921873.pt", map_location=device)
    ctx_dict = checkpoint['ctx_dict']
    ctx = GPTContext.from_dict(ctx_dict)
    ctx.dropout = dropout
    model = GPT2Model(ctx)
    model.load_state_dict(checkpoint['model_state'])

    model = torch.compile(model.to(device), fullgraph=True)

    if mode == "train":
        model.train()
    else:
        model.eval()

    losses = torch.zeros(256)
    for i in range(256):
        val_data = val_data_loader.next_batch()
        x = val_data[0].to(device)
        y = val_data[1].to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = model(x).logits
            loss = F.cross_entropy(output.view(-1, 50257), y.view(-1))
        losses[i] = loss.item()

    logging.info(f"mode={mode},dropout={dropout},loss={losses.mean().item()}")

# mode=train,dropout=0,loss=3.050893545150757
# check_val_loss("train", 0)

# mode=eval,dropout=0,loss=3.050893545150757
# check_val_loss("eval", 0)

# mode=train,dropout=0.1,loss=3.1256706714630127
# check_val_loss("train", 0.1)

# mode=eval,dropout=0.1,loss=3.050893545150757
# check_val_loss("eval", 0.1)