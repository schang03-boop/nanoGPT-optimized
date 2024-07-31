"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import numpy as np


def main():
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), 'enwik8.txt')

    with open(input_file_path, 'r') as f:
        data = f.read()
    utf8_data = data.encode('utf-8')
    print(f"length of UTF-8 dataset in characters: {len(utf8_data):,}")

    # get all the unique characters that occur in this text
    utf_bytes = sorted(list(set(utf8_data)))
    utf8_vocab_size = len(utf_bytes)
    print("all the unique bytes:", utf_bytes)
    print(f"vocab size: {utf8_vocab_size:,}")

    # Create a mapping from bytes to integers
    stoi = {ch: i for i, ch in enumerate(utf_bytes)}
    itos = {i: ch for i, ch in enumerate(utf_bytes)}

    # create the train and test splits
    n = len(utf8_data)
    train_data = utf8_data[:int(n * 0.9)]
    val_data = utf8_data[int(n * 0.9):int(n * 0.95)]
    test_data = utf8_data[int(n * 0.95):]

    # Encode both to integers
    train_ids = [stoi[b] for b in train_data]
    val_ids = [stoi[b] for b in val_data]
    test_ids = [stoi[b] for b in test_data]
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f"test has {len(test_ids):,} tokens")

    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))
    # Save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': utf8_vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)


if __name__ == '__main__':
    main()
# length of dataset in UTF-8 bytes:  100M
# all the unique characters:
#  re-order into 0-204
# vocab size: 205
# train has 90M tokens
# val has 5M tokens
# test has 5M tokens
