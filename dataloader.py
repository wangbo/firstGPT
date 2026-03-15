from tokenizer import SimpleTokenizer
import torch
import pickle


class SimpleDataloader(object):

    def __init__(self, dict_path, data_path, batch_size, block_size, device = "cpu"):
        self.dict_path = dict_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.block_size = block_size

        self.offset = 0
        self.step = batch_size * block_size
        self.token_array = None
        self.device = device

    def initialize(self):
        with open(self.data_path, "rb") as f:
            self.token_array = torch.tensor(pickle.load(f))
        self.total_token_num = len(self.token_array)

    def next_batch(self):
        if self.offset >= len(self.token_array):
            return None
        train_data = self.token_array[self.offset: self.offset + self.step].view(self.batch_size, self.block_size)
        labels = self.token_array[self.offset + 1 : self.offset + self.step + 1].view(self.batch_size, self.block_size)
        self.offset += self.step
        return train_data,labels

    def reset_offset(self):
        self.offset = 0


if __name__ == "__main__":
    dict_path = ""
    data_path = ""
    micro_size = 8
    block_size = 1024

    sd_loader = SimpleDataloader(dict_path, data_path, micro_size, block_size)
    sd_loader.initialize()

    tdata, labels = sd_loader.next_batch()
    print(tdata.shape == labels.shape)
    print(tdata[:, 1:1024] == labels[:, 0:1023])