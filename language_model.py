import tensorflow as tf
import numpy as np
from preprocess import *
from load_embeddings import *
import datetime
import os

deembed = False
use_word2vec = False #set to true for Experiment B
down_project = False #set to true for Experiment C

#load data and vocab
data = load_preprocessed_data()
vocab,inv_vocab = load_vocab()

#hyperparameters
hidden_size = 512
batch_size = 64
learning_rate = 0.0001
embedding_size = 100
n_train_steps = 300
number_checkpoint = 10
checkpoint_time = 20

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
    if use_word2vec: #non-trainable if using pretrained weights
        embedding_weights = tf.get_variable("embedding_weights", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
    else:
        embedding_weights = tf.get_variable("embedding_weights", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    #tensorflow function for embedding
    embedded_inputs = tf.nn.embedding_lookup(embedding_weights, inputs)

with tf.variable_scope("rnn"):
    #lstm cell
    if down_project:
        rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size*2, name="lstm_cell")
        initial_state = state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size*2]), tf.zeros([batch_size, hidden_size*2])) 
    else:
        rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, name="lstm_cell")
        initial_state = state = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, hidden_size]), tf.zeros([batch_size, hidden_size])) 
    #shared weights and bias for output layer
    if deembed:
        output_weights = tf.get_variable("output_weights", shape=[hidden_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
        output_bias = tf.get_variable("output_bias", shape=[embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    else:
        output_weights = tf.get_variable("output_weights", shape=[hidden_size, vocab_size], initializer=tf.contrib.layers.xavier_initializer())
        output_bias = tf.get_variable("output_bias", shape=[vocab_size], initializer=tf.contrib.layers.xavier_initializer())

    if down_project:
        down_project_weights = tf.get_variable("down_project_weights", shape=[hidden_size*2, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        down_project_bias = tf.get_variable("down_project_bias", shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())

    #fancy way of interating over time_steps which is middle dimension of embedded_inputs (batch_size x time_steps x embedding_size)
    output_list = []
    prediction_list = [] #not required for training
    embedded_inputs_unstacked = tf.unstack(embedded_inputs, axis=1)
    for embedded_input in embedded_inputs_unstacked:
        #calculate next state of rnn cell
        _, state = rnn(embedded_input, state)

        #calculate output layer
        if down_project:
            down_projected = tf.matmul(state.h, down_project_weights) + down_project_bias
            output = tf.matmul(down_projected, output_weights) + output_bias
        else:
            output = tf.matmul(state.h, output_weights) + output_bias
        #TODO "de-embed"???
        if deembed:
            output = tf.matmul(output, tf.transpose(embedding_weights))
        output_list += [output]
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
global_step = 0

#load word2vec
if use_word2vec:
    load_embedding(sess, vocab, embedding_weights, "./wordembeddings-dim100.word2vec", embedding_size, vocab_size)

#logging
log_dir = "./log/"+str(datetime.datetime.now().time())
writer = tf.summary.FileWriter(log_dir+"/summary", sess.graph)

#set up saving the  model
model_dir = os.path.abspath(os.path.join(log_dir, "checkpoints"))
model_prefix = os.path.join(model_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
saver = tf.train.Saver(tf.global_variables()) #, max_to_keep=number_checkpoint)

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
    global_step += 1
    x = y = batcher.get_batch(batch_size)
    l = train(x, y, i)
    if i % 1 == 0:
        print("{:.1f}% loss={}".format(100*i/n_train_steps, l))
    if (i+1) % checkpoint_time == 0:
        model_path = saver.save(sess, model_prefix, global_step=global_step)
        print("Model saved to {}\n".format(model_path))

model_path = saver.save(sess, model_prefix, global_step=global_step)
print("Final model saved to {}\n".format(model_path))
#test
#predicted = predict(np.reshape([data]*batch_size, [batch_size, input_n]))
#for i in range(input_n-1): 
#    print(predicted[:,i], data[i+1])

#logging
writer.close()

