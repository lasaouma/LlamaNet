import tensorflow as tf
import numpy as np
from preprocess import *
from perplexity import *
import argparse
import os

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='checkpoint_path', type=str)
parser.add_argument('--experiment', dest='experiment', type=str, default="A")
parser.add_argument('--data', dest='data_path', type=str, default="data/sentences.eval.preprocess")
args = parser.parse_args()
checkpoint_path = args.checkpoint_path
experiment = args.experiment

down_project = (experiment == "C")

#load data and vocab
vocab,inv_vocab = load_vocab()
sentences = load_preprocessed_data(data_path)

#hyperparameters
hidden_size = 512
batch_size = 1
embedding_size = 100

vocab_size = len(vocab)
time_steps = len(sentences[0])
input_n = time_steps
output_n = vocab_size

#CONSTRUCT NEURAL NETWORK
tf.reset_default_graph()

#placeholders for inputs and targets
with tf.variable_scope("inputs"):
    inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, input_n], name="inputs") 

with tf.variable_scope("embeddings"):
    embedding_weights = tf.get_variable("embedding_weights", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    #tensorflow function for embedding (under hood: convert input to one hot representation and multiply with embedding weight matrix)
    embedded_inputs = tf.nn.embedding_lookup(embedding_weights, inputs)

with tf.variable_scope("rnn"):
    if down_project:
    #lstm cell
        down_project_weights = tf.get_variable("down_project_weights", shape=[hidden_size*2, hidden_size], name="down_project_weights")
        rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size*2, name="lstm_cell")
        initial_state = state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size*2]), tf.zeros([batch_size, hidden_size*2])) 
    else:
        rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, name="lstm_cell")
        initial_state = state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size]), tf.zeros([batch_size, hidden_size])) 

    #shared weights and bias for output layer
    output_weights = tf.get_variable("output_weights", shape=[hidden_size, vocab_size], initializer=tf.contrib.layers.xavier_initializer())
    output_bias = tf.get_variable("output_bias", shape=[vocab_size], initializer=tf.contrib.layers.xavier_initializer())

    #fancy way of interating over time_steps which is middle dimension of embedded_inputs (batch_size x time_steps x embedding_size)
    output_list = []
    embedded_inputs_unstacked = tf.unstack(embedded_inputs, axis=1)

    state_list = []
    for embedded_input in embedded_inputs_unstacked:
        state_list += [state]
        #calculate next state of rnn cell
        _, state = rnn(embedded_input, state)

        #calculate output layer
        if down_project:
            down_projected = tf.matmul(state.h, down_project_weights)
            output = tf.matmul(down_projected, output_weights) + output_bias
        else:
            output = tf.matmul(state.h, output_weights) + output_bias
        output_list += [output]

    outputs = tf.stack(output_list, axis=1) # (batch_size x time_steps x embedding_size)
    softmax = tf.nn.softmax(outputs)

#initialise graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#load weights
saver = tf.train.Saver(var_list=[embedding_weights, output_weights, output_bias, down_project_weights, rnn.trainable_variables[0], rnn.trainable_variables[1]])
saver.restore(sess, checkpoint_path)

if not os.path.exists("./results"):
    os.makedirs("./results")

results_file = open("./results/group08.perplexity" + experiment, 'w')

written = 0
for sentence in sentences:
    prediction = sess.run(softmax, feed_dict={inputs:np.reshape(np.array(sentence), [1,-1])})[0]
    end_index = sentence.index(vocab['<eos>'])
   
    perplexity_score = perplexity(prediction[:end_index], sentence[1:end_index+1])

    results_file.write("{:.3f}\n".format(perplexity_score))

    if written % (len(sentences) / 10) == 0:
        print(perplexity_score)
    written += 1

results_file.close()

