import numpy as np
from huggingface_hub import snapshot_download
# import pandas as pd
import pyarrow.parquet as pq
from transformers import GPT2TokenizerFast
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import array

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# snapshot_download(
#     repo_id="HuggingFaceFW/fineweb-edu",
#     repo_type="dataset",
#     allow_patterns="sample/10BT/*",
#     local_dir="/home/git/fineweb10t",
#     max_workers=1
# )

data_dir = ""
output_dir = ""
file_list = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
file_list.sort()

# if os.path.exists(output_file_name):
#     os.remove(output_file_name)
#     logging.info(f"file already exists, drop it: {output_file_name}")
# else:
#     logging.info(f"file not exists, create it: {output_file_name}")

# token_limit = 2_500_000_000
text_column_name = "text"
read_parquet_batch_size = 1024
buffer_limit_bytes = 100 * 1024 * 1024 * 1 # 100M
dtype = np.uint16
max_threads = 16

def flush_buffer(f, token_buffer):
    buf_len = len(token_buffer)
    if buf_len == 0:
        return
    arr = np.asarray(token_buffer, dtype=dtype)
    arr.tofile(f)
    del arr
    del token_buffer[:]

def exec_encode(idx, input_file_path, output_file_path, output_file_name):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    eos_id = tokenizer.eos_token_id
    logging.info(f"begin idx={idx},"
                 f"input_path={input_file_path}, "
                 f"out_path={output_file_path}")
    # audit
    token_count = 0
    total_ufffd_count = 0
    # should not use python list here
    token_buffer = array.array('H')
    total_text_count = 0
    ufffd_text_count = 0
    with open(output_file_path, "wb") as f:
        pf = pq.ParquetFile(input_file_path)
        for rg_idx in range(pf.num_row_groups):
            for batch in pf.iter_batches(
                row_groups=[rg_idx],
                batch_size=read_parquet_batch_size,
                columns=[text_column_name],
            ):
                texts = batch.column(text_column_name)
                for i in range(len(texts)):
                    text = texts[i].as_py()
                    total_text_count += 1

                    ufffd_count = text.count('\ufffd')
                    if ufffd_count > 0:
                        ufffd_text_count += 1
                    total_ufffd_count += ufffd_count

                    text_token_ids = tokenizer.encode(text)
                    text_token_ids.append(eos_id)

                    token_count += len(text_token_ids)
                    token_buffer.extend(text_token_ids)
                    buf_bytes = len(token_buffer) * 2

                    if buf_bytes > buffer_limit_bytes:
                        logging.info(f"idx={idx} flush,token_count={token_count},"
                                     f"ufffd_count={total_ufffd_count},"
                                     f"ufd_rate={total_ufffd_count/token_count*100:.2f}%")
                        flush_buffer(f,token_buffer)

        flush_buffer(f,token_buffer)
    rate = total_ufffd_count / token_count * 100 if token_count > 0 else 0
    logging.info(f"finished idx={idx},f_name={output_file_name},"
                 f"token_count={token_count},"
                 f"ufffd_count={total_ufffd_count},"
                 f"total_text_count={total_text_count},"
                 f"ufffd_text_count={ufffd_text_count},"
                 f"ufd_rate={rate:.2f}%")

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = {}
    for idx, input_file_name in enumerate(file_list):
        input_file_path = os.path.join(data_dir, input_file_name)

        output_file_name = input_file_name.split('.')[0] + ".bin"
        output_file_path = os.path.join(output_dir, output_file_name)

        future = executor.submit(exec_encode, idx, input_file_path, output_file_path, output_file_name)
        futures[future] = output_file_name
        logging.info(f"submit idx={idx},name={input_file_name}")

    for future in as_completed(futures):
        output_file_name = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"{output_file_name} failed: {e}")

