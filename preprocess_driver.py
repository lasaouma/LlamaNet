import preprocess

#head -n 50000 sentences.train > sentences.test

preprocess.preprocess_data("sentences.test", write_file="data/sentences.train.preprocess", vocab_size=20000)

vocab,inv_vocab = preprocess.load_vocab()
data = preprocess.load_preprocessed_data()

eval_data = preprocess.preprocess_eval_data(vocab, read_file="sentences.eval", write_file="data/sentences.eval.preprocess")
eval_data_read = load_preprocessed_data(read_file="data/sentences.eval.preprocess")

print("---preprocessed data---")
print("eval file good: " + str(eval_data == eval_data_read))
print("vocab len=" + str(len(vocab)))
print("sentences num=" + str(len(data)))
print("sentence len=" + str(len(data[0])))
print("eval num=" + str(len(eval_data)))
print("eval len=" + str(len(eval_data[0])))
