import tensorflow as tf
import numpy as np
from preprocess import *

n=20 #generated sentence max length

vocab,inv_vocab = load_vocab()
data = load_continuation_data()

hidden_size = 512
vocab_size = len(vocab)
embedding_size = 100

#construct neural network graph
embedding_weights = tf.get_variable("embeddings/embedding_weights", shape=[vocab_size, embedding_size])
output_weights = tf.get_variable("rnn/output_weights", shape=[hidden_size, vocab_size])
output_bias = tf.get_variable("rnn/output_bias", shape=[vocab_size])
rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, name="rnn/lstm_cell")
input_ = tf.placeholder(shape=[], dtype=tf.int32, name="input")
embedded_input = tf.reshape(tf.nn.embedding_lookup(embedding_weights, input_), [1,-1])
input_state = tf.placeholder(shape=[1,hidden_size], dtype=tf.float32, name="input_state")
input_state_proper = tf.nn.rnn_cell.LSTMStateTuple(input_state, input_state)
_, output_state = rnn(embedded_input, input_state_proper)
output = tf.matmul(output_state.h, output_weights) + output_bias
output_softmax = tf.nn.softmax(output)
#prediction = tf.argmax(output,axis=1)

sess = tf.Session()

#load weights
saver = tf.train.Saver(var_list=[embedding_weights, output_weights, output_bias, rnn.trainable_variables[0], rnn.trainable_variables[1]])
saver.restore(sess, "/home/xan/zur/nlu/pro/LlamaNet/log/23:39:39.662477/checkpoints/model-120") #TODO automatically locate

def predict(inp, inp_state):
    #sample words based on softmax distribution TODO supposed to use argmax?
    softmax,out_state = sess.run([output_softmax,output_state.h], feed_dict={input_: inp, input_state: np.reshape(inp_state,[1,-1])})
    predict = np.random.choice(range(vocab_size), p=softmax[0])
    return predict,out_state

count = 0 #debug
output_sentences = []
for sentence in data:
    i = 0
    state_ = np.zeros([hidden_size])
    _, state_ = predict(2, state_) #2=<bos>
    output_sentence = []
    for word in sentence:
        #predict word
        predicted_word, state_ = predict(word, state_)
        output_sentence += [inv_vocab[word]]
        i += 1
    while i < n and predicted_word != 1: #not <eos>
        new_predicted_word, new_state = predict(predicted_word, state_)
        if new_predicted_word not in [0,2,3]: #not <unk>, <bos>, <pad>
            predicted_word = new_predicted_word
            state_ = new_state
            output_sentence += [inv_vocab[predicted_word]]
            i += 1
    count += 1 #debug
    if count % 100 == 0: #debug
        print_str = "" #debug
        for w in output_sentence: #debug
            print_str += (w + " ") #debug
        print(print_str) #debug
    output_sentences += [output_sentence]

print(output_sentences)
