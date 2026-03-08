from tokenizer import SimpleTokenizer
import os
import json
import pickle


if __name__ == "__main__":
    file_path = ""
    dict_path = ""
    token_file_path = ""

    build_vocab = False
    simple_tk = SimpleTokenizer()

    if build_vocab:
        simple_tk.build_vocab(file_path)
        simple_tk.save_vocab(dict_path)
    else:
        simple_tk.load_vocab(dict_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"data char num:{len(text)}")
    token_id_arr = simple_tk.encode(text)
    print(f"token num:{len(token_id_arr)}")

    if os.path.isfile(token_file_path):
        print(f"file exists:{token_file_path}")
        exit(0)

    with open(token_file_path, "wb") as f:
        pickle.dump(token_id_arr, f)

