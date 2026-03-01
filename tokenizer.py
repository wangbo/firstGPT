import json
import os
import torch

class SimpleTokenizer:

    def __init__(self):
        self.str_to_i = {}
        self.i_to_str = {}
        self.vocab_size = 0
        self.total_token_num = 0

    def make_token_dict_from_single_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.total_token_num = len(text)
        print(f"total tokens:{self.total_token_num}")

        distinct_chars = sorted(list(set(text)))
        self.vocab_size = len(distinct_chars)

        print(f"vocab size:{self.vocab_size}")

        self.str_to_i = {ch: i for i, ch in enumerate(distinct_chars)}
        self.i_to_str = {i: ch for i, ch in enumerate(distinct_chars)}

    def save_dict_to_disk(self, dict_path):
        final_dict = {
            "stoi" : self.str_to_i,
            "itos" : self.i_to_str
        }

        if os.path.isfile(dict_path):
            print(f"file exists:{dict_path}")
            return

        with open(dict_path, 'w', encoding='utf-8') as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=4)


    def load_token_dict(self, dict_path):
        if not os.path.exists(dict_path):
            print(f"file not exists: {dict_path}")
            return

        with open(dict_path, 'r', encoding='utf-8') as f:
            merged_dict = json.load(f)
            for k,v in merged_dict['stoi'].items():
                self.str_to_i[str(k)] = int(v)

            for k,v in merged_dict['itos'].items():
                self.i_to_str[int(k)] = v

        self.vocab_size = len(self.str_to_i)

    def load_data(self, data_file_path):
        with open(data_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        self.total_token_num = len(text)

        return train_data, val_data

    def encode(self, input_str):
        return [self.str_to_i[ch] for ch in input_str]

    def decode(self, input_token_ids):
        ret = [self.i_to_str[i] for i in input_token_ids]
        return ''.join(ret)

if __name__ == "__main__":
    file_name = "all"
    file_path = "/home/wangbo/git/cn_data/" + file_name + ".txt"
    dict_path = "/home/wangbo/git/cn_data/" + file_name + ".dict"

    simple_tk = SimpleTokenizer()
    # simple_tk.make_token_dict_from_single_file(file_path)
    # simple_tk.save_dict_to_disk(dict_path)

    simple_tk.load_token_dict(dict_path)
    ret = simple_tk.encode("西游记")
    print(ret)
    print(simple_tk.decode(ret))
