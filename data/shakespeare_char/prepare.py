"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import numpy as np


def create_word_boundaries(encoded_data, byte_to_idx, idx_to_byte):
    # Convert encoded_data to NumPy array if it's not already
    encoded_data = np.array(encoded_data, dtype=np.uint16)

    # Create arrays for current and previous encoded bytes
    current_encoded = encoded_data
    space_idx = byte_to_idx[' '.encode('utf-8')[0]]
    prev_encoded = np.pad(encoded_data[:-1], (1, 0), constant_values=space_idx)

    # Function to check if a byte is the start of a UTF-8 character
    def is_start_of_char(byte_idx):
        return (idx_to_byte[byte_idx] & 0xC0) != 0x80

    # Function to check if a byte is a space or punctuation
    def is_space_or_punct(byte_idx):
        return idx_to_byte[byte_idx] in b' \t\n.,!?;:()[]{}"\''

    # Vectorize the functions
    v_is_start_of_char = np.vectorize(is_start_of_char)
    v_is_space_or_punct = np.vectorize(is_space_or_punct)

    # Identify start of UTF-8 characters
    is_start_of_char = v_is_start_of_char(current_encoded)

    # Identify space or punctuation
    is_space_or_punct = v_is_space_or_punct(current_encoded)

    # Identify word boundaries
    is_word_boundary = is_start_of_char & (
            is_space_or_punct |
            v_is_space_or_punct(prev_encoded) |
            (np.vectorize(lambda x: idx_to_byte[x] < 0x80)(prev_encoded))
    )

    return is_word_boundary.astype(np.uint8)


def main():
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), 'enwik8.txt')

    with open(input_file_path, 'r') as f:
        data = f.read()
    utf8_data = data.encode('utf-8')[:30]
    print(f"length of dataset in characters: {len(data):,}")
    print(f"length of UTF-8 dataset in characters: {len(utf8_data):,}")

    # get all the unique characters that occur in this text
    utf_bytes = sorted(list(set(utf8_data)))
    utf8_vocab_size = len(utf_bytes)
    print("all the unique bytes:", utf_bytes)
    print(f"vocab size: {utf8_vocab_size:,}")

    # Create a mapping from bytes to integers
    byte_to_idx = {ch: i for i, ch in enumerate(utf_bytes)}
    idx_to_byte = {i: ch for i, ch in enumerate(utf_bytes)}

    encoded_data = np.array([byte_to_idx[b] for b in utf8_data], dtype=np.uint16)
    # Create word boundary data
    word_boundaries = create_word_boundaries(encoded_data, byte_to_idx, idx_to_byte)

    # create the train and test splits
    n = len(utf8_data)
    train_data = encoded_data[:int(n * 0.9)]
    val_data = encoded_data[int(n * 0.9):int(n * 0.95)]
    test_data = encoded_data[int(n * 0.95):]
    print(f"train has {len(train_data):,} tokens")
    print(f"val has {len(val_data):,} tokens")
    print(f"test has {len(test_data):,} tokens")

    train_word_boundaries = word_boundaries[:int(n * 0.9)]
    val_word_boundaries = word_boundaries[int(n * 0.9):int(n * 0.95)]
    test_word_boundaries = word_boundaries[int(n * 0.95):]

    # Export to bin files
    train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    test_data.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

    train_word_boundaries.tofile(os.path.join(os.path.dirname(__file__), 'train_word_boundaries.bin'))
    val_word_boundaries.tofile(os.path.join(os.path.dirname(__file__), 'val_word_boundaries.bin'))
    test_word_boundaries.tofile(os.path.join(os.path.dirname(__file__), 'test_word_boundaries.bin'))
    # Save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': utf8_vocab_size,
        'idx_to_byte': idx_to_byte,
        'byte_to_idx': byte_to_idx,
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
