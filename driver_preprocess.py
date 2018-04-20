import preprocess

preprocess.preprocess_data(read_file="data/sentences.train", write_file="data/sentences.train.preprocess", vocab_size=20000)

vocab,inv_vocab = preprocess.load_vocab()
data = preprocess.load_preprocessed_data()

preprocess.preprocess_eval_data(vocab, read_file="data/sentences.test", write_file="data/sentences.test.preprocess")

eval_data = preprocess.load_preprocessed_data(read_file="data/sentences.test.preprocess")

print("[Preprocessing done]")
print("vocab len=" + str(len(vocab)))
print("sentences num=" + str(len(data)))
print("sentence len=" + str(len(data[0])))
print("test num=" + str(len(eval_data)))
print("test len=" + str(len(eval_data[0])))
