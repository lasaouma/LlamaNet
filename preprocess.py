import pickle
import numpy as np
import os

def write_vocab(id_word, word_id):
    if not os.path.exists("./vocab"):
        os.makedirs("./vocab")
    with open("./vocab/WordID.pkl", "wb") as f:
        pickle.dump(word_id, f)
    with open("./vocab/IDWord.pkl", "wb") as f:
        pickle.dump(id_word, f)

def load_vocab():
    with open('./vocab/IDWord.pkl', 'rb') as IDWord_file:
        id_word = pickle.load(IDWord_file)
    with open('./vocab/WordID.pkl', 'rb') as WordID_file:
        word_id = pickle.load(WordID_file)

    return word_id, id_word

def preprocess_data(read_file, write_file="sentences.train.preprocess", vocab_size=20000, line_len=30):
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

    # build dictonary of word to ids
    word_id = {word: i for i, word in enumerate(known_words, 4)}
    word_id["<unk>"] = 0
    word_id["<eos>"] = 1
    word_id["<bos>"] = 2
    word_id["<pad>"] = 3

    id_word = dict(zip(word_id.values(), word_id.keys()))  # reverse dict to get word from id

    # write to pickle
    write_vocab(id_word, word_id)

    known_words.extend(["<eos>", "<bos>", "<pad>"])

    processed_lines = []
    if write_file is not None:
        write_file = open(write_file, 'w')

    for line in lines:
        line = ['<bos>'] + line
        line += ['<eos>']
        line += ['<pad>']*(line_len-len(line))
        
        # exchanges words with ids and replaces words that are not in vocab with the id of unk
        for idx, word in enumerate(line):
            if word not in known_words:
                line[idx] = word_id['<unk>']
            else:
                line[idx] = word_id[word]
        

        processed_lines.append(list(line))

        if write_file is not None:
            line_str1 = ' '.join(str(x) for x in line)
            line_str1 += '\n'
            write_file.write(line_str1)

    if write_file is not None:
        write_file.close()

    lines_np = np.array(processed_lines)
    return lines_np,word_id,id_word

def preprocess_eval_data(vocab, read_file, write_file="sentences.eval.preprocess", line_len=30):
    # read lines
    lines = []
    read_file = open(read_file, 'r')
    for line in read_file:
        split_line = line[:-1].split(' ')
        if len(split_line) <= line_len - 2:
            lines += [split_line]
    read_file.close()

    processed_lines = []
    if write_file is not None: # To be consistent with a single format, lets put the folder out
        write_file = open(write_file, 'w')

    for line in lines:
        line = ['<bos>'] + line
        line += ['<eos>']
        line += ['<pad>']*(line_len-len(line))
        
        # exchanges words with ids and replaces words that are not in vocab with the id of unk
        for idx, word in enumerate(line):
            if word not in vocab.keys():
                line[idx] = vocab['<unk>']
            else:
                line[idx] = vocab[word]
        
        processed_lines.append(list(line))

        if write_file is not None:
            line_str1 = ' '.join(str(x) for x in line)
            line_str1 += '\n'
            write_file.write(line_str1)

    if write_file is not None:
        write_file.close()

    lines_np = np.array(processed_lines)
    return lines_np

def load_continuation_data(continuation_path='./data/sentences.continuation', sentence_lenght=20):
    word_id, id_word = load_vocab()
    vocab = list(word_id.keys())

    continuation = []
    with open(continuation_path, "r") as continuation_sentences:
        for sentence in continuation_sentences:
            words = sentence.strip().split(" ")

            if len(words) < sentence_lenght:
                for idx, word in enumerate(words):
                    if word not in vocab:
                        words[idx] = word_id['<unk>']
                    else:
                        words[idx] = word_id[word]

                continuation.append(words)
    return np.array(continuation)

# read data from preprocessed file written by preprocess_data
def load_preprocessed_data(read_file="./data/sentences.preprocess"):
    lines = []
    read_file = open(read_file, 'r')
    for line in read_file:
        lines += [[int(word) for word in line[:-1].split(' ')]]
    read_file.close()
    return lines
