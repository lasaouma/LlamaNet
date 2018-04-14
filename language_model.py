import tensorflow as tf
import numpy as np
from preprocess import *

#load data and vocab
data = load_preprocessed_data()
vocab,inv_vocab = load_vocab()

#hyperparameters
hidden_size = 32
batch_size = 32
learning_rate = 0.0001
embedding_size = 16
n_train_steps = 100000

vocab_size = len(vocab)
time_steps = len(data[0])
input_n = time_steps
output_n = vocab_size

class Batcher: #TODO
    def __init__(self, data):
        self.data = np.array(data)
        self.data_size = len(data)
        self.current_index = 0
        self.shuffle()

    def shuffle(self):
        shuffle_indices = np.random.permutation(np.arange(self.data_size))
        self.shuffled_data = self.data[shuffle_indices]

    def get_batch(self, batch_size):
        if self.current_index+batch_size >= self.data_size:
            self.current_index = 0
            self.shuffle()
        batch_data = self.shuffled_data[self.current_index:self.current_index+batch_size]
        self.current_index += batch_size
        return batch_data

#construct neural network #TODO seemingly doesn't work
tf.reset_default_graph()
        
with tf.variable_scope("inputs"):
    inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, input_n], name="inputs") 
    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, input_n], name="targets")
    targets_one_hot = tf.one_hot(targets, vocab_size)

with tf.variable_scope("embeddings"):
    embedding_weights = tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
    inputs_one_hot = tf.one_hot(inputs, vocab_size)
    embedded_inputs = tf.nn.embedding_lookup(embedding_weights, inputs)
    #mbedded_inputs = inputs_one_hot

with tf.variable_scope("rnn"):

    rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
    initial_state = state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size]), tf.zeros([batch_size, hidden_size])) 
    output_weights = tf.random_uniform([hidden_size, embedding_size], -1.0, 1.0)
    output_bias = tf.random_uniform([embedding_size], -1.0, 1.0)

    output_list = []
    prediction_list = []
    embedded_inputs_unstacked = tf.unstack(embedded_inputs, axis=1)
    for embedded_input in embedded_inputs_unstacked:
        _, state = rnn(embedded_input, state)
        output = tf.matmul(state.h, output_weights) + output_bias
        output_decoded = tf.matmul(output, tf.transpose(embedding_weights))
        output_list += [output_decoded]

        prediction_list += [tf.argmax(output,axis=1)]
    outputs = tf.stack(output_list, axis=1)
    predictions = tf.stack(prediction_list, axis=1)

with tf.variable_scope("optimizer"):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs[:,:-1,:], labels=targets_one_hot[:,1:,:]) #correct?
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def train(x, y):
    l,_ = sess.run([loss, optimizer], feed_dict={inputs: x, targets: y})
    return l

def predict(x):
    return sess.run(predictions, feed_dict={inputs: x})



#DRIVER

batcher = Batcher(data)

#train
for i in range(n_train_steps):
    x = y = batcher.get_batch(batch_size)
    l = train(x, y)
    if i % 1000 == 0:
        print("{:.1f}% loss={}".format(100*i/n_train_steps, l[0][0]))

#test
#predicted = predict(np.reshape([data]*batch_size, [batch_size, input_n]))
#for i in range(input_n-1): 
#    print(predicted[:,i], data[i+1])

#logging
writer = tf.summary.FileWriter("./log", sess.graph)
writer.close()

