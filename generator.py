import tensorflow as tf
import numpy as np
from preprocess import *

n=20 #generated sentence max length

vocab,inv_vocab = load_vocab()
data = #... TODO load data

hidden_size = 512
vocab_size = len(vocab)
embedding_size = 100

#TODO load neural network weights (embedding, output, rnn)
embedding_weights = tf.get_variable("output_weights", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
output_weights = tf.get_variable("output_weights", shape=[hidden_size, vocab_size], initializer=tf.contrib.layers.xavier_initializer())
output_bias = tf.get_variable("output_bias", shape=[vocab_size], initializer=tf.contrib.layers.xavier_initializer())
rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
input_ = tf.Placeholder(dtype=tf.float32)
embedded_input = tf.nn.embedding_lookup(embedding_weights, input_)
input_state = tf.Placeholder([hidden_size], dtype=tf.float32)
input_state_proper = tf.nn.rnn_cell.LSTMStateTuple(input_state, input_state)
_, output_state = rnn(embedded_input, input_state_proper)
output = tf.matmul(output_state.h, output_weights) + output_bias
prediction = tf.argmax(output)

sess = tf.Session()

def predict(inp, inp_state):
    predict,out_state = sess.run([prediction,output_state], feed_dict={input_: inp, input_state: inp_state})
    return inv_vocab(predict),out_state

for sentence in sentences:
    i = 0
    state_ = np.zeros([hidden_size])
    _, state_ = predict(2, state_) #2=<bos>
    output_sentence = []
    for word in sentence:
        #predict word
        predicted_word, state_ = predict(word, state_)
        output_sentence += [word]
        i += 1
    while i < n and predicted_word != "<eos>":
        predicted_word = predict(predicted_word, state_)
        output_sentence += [predicted_word]
    output_sentences += [output_sentence]

print(output_sentences)
