import numpy as np
from huggingface_hub import snapshot_download
# import pandas as pd
import pyarrow.parquet as pq
from transformers import GPT2TokenizerFast
import os

# snapshot_download(
#     repo_id="HuggingFaceFW/fineweb-edu",
#     repo_type="dataset",
#     allow_patterns="sample/10BT/*",
#     local_dir="/home/git/fineweb10t",
#     max_workers=1
# )

# file_name = ""
output_file_name = ""

data_dir = ""
file_list = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
file_list.sort()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",local_files_only=True)

if os.path.exists(output_file_name):
    os.remove(output_file_name)
    print(f"file already exists, drop it: {output_file_name}")
else:
    print(f"file not exists, create it: {output_file_name}")

# token_limit = 2_500_000_000
token_limit = 100000000
text_column_name = "text"
read_parquet_batch_size = 1024
eos_id = tokenizer.eos_token_id
buffer_tokens = 100 * 1024 * 1024 # 600M bytes
dtype = np.uint16

# audit
batch_count = 0
text_count = 0
token_count = 0

token_buffer = []

def flush_buffer(f):
    buf_len = len(token_buffer)
    if buf_len == 0:
        return
    arr = np.asarray(token_buffer, dtype=dtype)
    arr.tofile(f)
    token_buffer.clear()
    print(f"flush buffer: token count:{buf_len}, total token count:{token_count}")

stop_flag = False
with open(output_file_name, "wb") as f:
    for file_name in file_list:
        if stop_flag:
            break
        data_path = os.path.join(data_dir, file_name)
        pf = pq.ParquetFile(data_path)
        total_row_groups = pf.num_row_groups
        print(f"read file: {file_name}, "
              f"token count: {token_count}, "
              f"token progress:{token_count / token_limit * 100:.2f}% ")
        for rg_idx in range(pf.num_row_groups):
            print(f"process: {rg_idx}/{total_row_groups}, "
                  f"token count: {token_count}, "
                  f"token progress:{token_count/token_limit * 100:.2f}% ")

            if token_count > token_limit:
                stop_flag = True
                break

            for batch in pf.iter_batches(
                row_groups=[rg_idx],
                batch_size=read_parquet_batch_size,
                columns=[text_column_name],
            ):
                batch_count += 1
                texts = batch.column(text_column_name)
                for i in range(len(texts)):
                    text_count += 1
                    text = texts[i].as_py()
                    text_token_ids = tokenizer.encode(text)
                    text_token_ids.append(eos_id)

                    token_count += len(text_token_ids)
                    token_buffer.extend(text_token_ids)

                    if len(token_buffer) >= buffer_tokens:
                        flush_buffer(f)
    flush_buffer(f)

print(f"final audit: total token count:{token_count}, batch count:{batch_count} , text count:{text_count}.")

