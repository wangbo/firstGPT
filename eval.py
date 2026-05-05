from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm

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
gpt2_raw_model = AutoModelForCausalLM.from_pretrained(
    model_name,local_files_only=True).to(device)

eos_token_id = tokenizer.eos_token_id

# ppl is exp for forward loss
@torch.no_grad()
def eval_ppl(token_list, model):
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
    logging.info(f"token size:{len(token_list)}")

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

# ppl = eval_ppl(token_list, gpt2_raw_model)
# logging.info(ppl)
# 固定窗口计算出来的值 tensor(46.4863
# 滑动窗口步长为512时计算出来的值 tensor(42.6428

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

# correct count:2967,rate: 29.55%
eval_hella_swag(gpt2_raw_model)

            
