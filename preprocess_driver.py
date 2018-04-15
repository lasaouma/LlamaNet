import preprocess

#head -n 50000 sentences.train > sentences.test

preprocess.preprocess_data("dataset/sentences.train", vocab_size=20000)

vocab,inv_vocab = preprocess.load_vocab()
data = preprocess.load_preprocessed_data()

print("---preprocessed data---")
print("vocab len=" + str(len(vocab)))
print("sentences num=" + str(len(data)))
print("sentence len=" + str(len(data[0])))
