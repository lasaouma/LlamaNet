import tensorflow as tf
import numpy as np
from preprocess import *

#load data and vocab
data = load_preprocessed_data()
vocab,inv_vocab = load_vocab()

#hyperparameters
hidden_size = 512
batch_size = 64
learning_rate = 0.0001
embedding_size = 100
n_train_steps = 100

vocab_size = len(vocab)
time_steps = len(data[0])
input_n = time_steps
output_n = vocab_size

#batcher for getting shuffled training samples
class Batcher:
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

#CONSTRUCT NEURAL NETWORK
tf.reset_default_graph()

#placeholders for inputs and targets
with tf.variable_scope("inputs"):
    inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, input_n], name="inputs") 
    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, input_n], name="targets")

#get embedding vectors
with tf.variable_scope("embeddings"):
    #embedding weight matrix
    embedding_weights = tf.get_variable("embedding_weights", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    #transform inputs to one hot vectors
    inputs_one_hot = tf.one_hot(inputs, vocab_size)
    #tensorflow function for embedding
    embedded_inputs = tf.nn.embedding_lookup(embedding_weights, inputs)

with tf.variable_scope("rnn"):
    #lstm cell
    rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
    #state of lstm cell
    initial_state = state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size]), tf.zeros([batch_size, hidden_size])) 
    #shared weights and bias for output layer ("softmax")
    output_weights = tf.get_variable("output_weights", shape=[hidden_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    output_bias = tf.get_variable("output_bias", shape=[embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    
    #fancy way of interating over time_steps which is middle dimension of embedded_inputs (batch_size x time_steps x embedding_size)
    output_list = []
    prediction_list = [] #not required for training
    embedded_inputs_unstacked = tf.unstack(embedded_inputs, axis=1)
    for embedded_input in embedded_inputs_unstacked:
        #calculate next state of rnn cell
        _, state = rnn(embedded_input, state)
        #calculate output layer ("softmax")
        output = tf.matmul(state.h, output_weights) + output_bias
        #"de-embed" softmax ???
        output_decoded = tf.matmul(output, tf.transpose(embedding_weights))
        output_list += [output_decoded]
        prediction_list += [tf.argmax(output,axis=1)] #not required for training
    outputs = tf.stack(output_list, axis=1) # (batch_size x time_steps x embedding_size
    predictions = tf.stack(prediction_list, axis=1) #not required for training

with tf.variable_scope("optimizer"):
    #calculate loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs[:,:-1,:], labels=targets[:,1:])) #TODO take mean???
    #define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #clip gradients
    var = tf.trainable_variables()
    grad,_ = tf.clip_by_global_norm(tf.gradients(loss, var), 5)
    #training operation
    train_op = optimizer.apply_gradients(zip(grad,var))
    #logging
    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()

#initialise graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#logging
writer = tf.summary.FileWriter("./log", sess.graph)

#perform training step and return loss
def train(x, y, step):
    l,m,_ = sess.run([loss, merged, train_op], feed_dict={inputs: x, targets: y}) #train
    writer.add_summary(m, step) #add (step,loss) to summary for visualising loss 
    return l

def predict(x): #not required for training
    return sess.run(predictions, feed_dict={inputs: x})

#DRIVER

batcher = Batcher(data)

#train
for i in range(n_train_steps):
    x = y = batcher.get_batch(batch_size)
    l = train(x, y, i)
    if i % 1 == 0:
        print("{:.1f}% loss={}".format(100*i/n_train_steps, l))

#test
#predicted = predict(np.reshape([data]*batch_size, [batch_size, input_n]))
#for i in range(input_n-1): 
#    print(predicted[:,i], data[i+1])

#logging
writer.close()

