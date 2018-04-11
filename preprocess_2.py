import pickle
import numpy as np
import os


def preprocess_data(read_file, write_file=None, vocab_size=20000, line_len=30):
    # read lines and construct vocabulary
    vocab = dict()
    lines = []
    read_file = open(read_file, 'r')
    for line in read_file:
        split_line = line[:-1].split(' ')
        for word in split_line:
            if word in vocab.keys():
                vocab[word] += 1
            else:
                vocab[word] = 1
        if len(split_line) <= line_len - 2:
            lines += [split_line]
    read_file.close()

    # add most frequent words in vocabulary to known words
    known_words = []
    for word in sorted(vocab, key=vocab.get, reverse=True):
        if len(known_words) < vocab_size - 4:
            known_words += [word]

    # build dictonary of word to IDs
    word_ID = {word: i for i, word in enumerate(known_words, 4)}
    word_ID["<unk>"] = 0
    word_ID["<eos>"] = 1
    word_ID["<bos>"] = 2
    word_ID["<pad>"] = 4

    ID_word = dict(zip(word_ID.values(), word_ID.keys()))  # reverse dict to get word from ID

    # write to pickle
    if not os.path.exists("./vocab"):
        os.makedirs("./vocab")
    with open("./vocab/wordID.pkl", "wb") as f:
        pickle.dump(word_ID, f)
    with open("./vocab/IDWord.pkl", "wb") as f:
        pickle.dump(ID_word, f)

    known_words.extend(["<eos>", "<bos>", "<pad>"])

    processed_lines = []
    if write_file is not None:
        write_file = open(write_file, 'w')

    for line in lines:
        line.extend(['<pad>'] * ((line_len-2) - len(line)))
        line.insert(0, '<eos>')
        line.insert(len(line), '<bos>')

        # exchanges words with ids and replaces words that are not in vocab with the id of unk
        for idx, word in enumerate(line):
            if word not in known_words:
                line[idx] = word_ID['<unk>']
            else:
                line[idx] = word_ID[word]

        processed_lines.append(list(line))

        if write_file is not None:
            line_str1 = ' '.join(str(x) for x in line)
            line_str1 += '\n'
            write_file.write(line_str1)

    if write_file is not None:
        write_file.close()

    lines_np = np.array(processed_lines)
    return lines_np


# read data from preprocessed file written by preprocess_data
def read_preprocessed_data(read_file):
    lines = []
    read_file = open(read_file, 'r')
    for line in read_file:
        lines += [line[:-1].split(' ')]
    read_file.close()
    return lines
