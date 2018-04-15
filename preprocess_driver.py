import preprocess

#head -n 50000 sentences.train > sentences.test

preprocess.preprocess_data("sentences.test", vocab_size=5000)

vocab,inv_vocab = preprocess.load_vocab()
data = preprocess.load_preprocessed_data()

print("---preprocessed data---")
print("vocab len=" + str(len(vocab)))
print("sentences num=" + str(len(data)))
print("sentence len=" + str(len(data[0])))
