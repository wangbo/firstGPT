import torch
from transformers import GPT2TokenizerFast
import numpy as np


class SimpleDataloader(object):

    def __init__(self, token_arr, start_idx, end_idx,
                 batch_size, block_size):
        assert start_idx >= 0
        assert start_idx < end_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.token_arr = token_arr
        assert self.end_idx < len(self.token_arr)

        self.batch_size = batch_size
        self.block_size = block_size
        self.step = batch_size * block_size

    def next_batch(self):
        np_x = self.token_arr[self.start_idx : self.start_idx + self.step].astype(np.int64)
        np_y = self.token_arr[self.start_idx + 1 : self.start_idx + self.step + 1].astype(np.int64)

        x = torch.from_numpy(np_x).view(self.batch_size, self.block_size)
        y = torch.from_numpy(np_y).view(self.batch_size, self.block_size)
        self.start_idx += self.step
        return x, y

    def token_num(self):
        return self.end_idx - self.start_idx + 1

    def reset_start_idx(self):
        self.start_idx = 0

if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    print(tokenizer.vocab_size)