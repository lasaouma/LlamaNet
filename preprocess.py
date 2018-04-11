def preprocess_data(read_file, write_file=None, vocab_size=20000, line_len=30):
    #read lines and construct vocabulary
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
        if len(split_line) <= line_len-2:
            lines += [split_line]
    read_file.close()

    #add most frequent words in vocabulary to known words
    known_words = []
    for word in sorted(vocab, key=vocab.get, reverse=True):
        if len(known_words) < vocab_size:
            known_words += [word]

    #add tokens to each line
    processed_lines = []
    if write_file is not None:
        write_file = open(write_file, 'w')
    for line in lines:
        processed_line = "<bos> "
        for word in line:
            if word in known_words:
                processed_line += word
            else:
                processed_line += "<unk>"
            processed_line += " "
        for i in range(line_len-2 - len(line)):
            processed_line += "<pad> "
        processed_line += "<eos>\n"
        processed_lines += [processed_line[:-1].split(' ')]
        if write_file is not None:
            write_file.write(processed_line)
    if write_file is not None:
        write_file.close()

    return processed_lines

#read data from preprocessed file written by preprocess_data
def read_preprocessed_data(read_file):
    lines = []
    read_file = open(read_file, 'r')
    for line in read_file:
        lines += [line[:-1].split(' ')]
    read_file.close()
    return lines 
