import tensorflow as tf
import numpy as np
from preprocess import *
from load_embeddings import *
from batcher import *
import datetime
import os

#LOAD DATA AND SET PARAMETERS
use_word2vec = False #set to true for Experiment B
down_project = False #set to true for Experiment C

#load data and vocab
data = load_preprocessed_data("data/sentences.train.preprocess")
vocab,inv_vocab = load_vocab() # Same vocab for both

#hyperparameters
hidden_size = 512
batch_size = 64
learning_rate = 1e-4 
embedding_size = 100
n_train_steps = 500000
number_checkpoint = 10
checkpoint_time = 50

vocab_size = len(vocab)
time_steps = len(data[0])
input_n = time_steps
output_n = vocab_size

#CONSTRUCT NEURAL NETWORK
tf.reset_default_graph()

#placeholders for inputs and targets
with tf.variable_scope("inputs"):
    inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, input_n], name="inputs") 
with tf.variable_scope("targets"):
    targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, input_n], name="targets")

with tf.variable_scope("embeddings"):
    embedding_weights = tf.get_variable("embedding_weights", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    #tensorflow function for embedding (under hood: convert input to one hot representation and multiply with embedding weight matrix)
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
    output_weights = tf.get_variable("output_weights", shape=[hidden_size, vocab_size], initializer=tf.contrib.layers.xavier_initializer())
    output_bias = tf.get_variable("output_bias", shape=[vocab_size], initializer=tf.contrib.layers.xavier_initializer())

    if down_project: #experiment C
        down_project_weights = tf.get_variable("down_project_weights", shape=[hidden_size*2, hidden_size], initializer=tf.contrib.layers.xavier_initializer())

    #fancy way of interating over time_steps which is middle dimension of embedded_inputs (batch_size x time_steps x embedding_size)
    output_list = []
    prediction_list = []
    softmax_list = []
    embedded_inputs_unstacked = tf.unstack(embedded_inputs, axis=1)

    for embedded_input in embedded_inputs_unstacked:
        #calculate next state of rnn cell
        _, state = rnn(embedded_input, state)

        #calculate output layer
        if down_project: #experiment C
            down_projected = tf.matmul(state.h, down_project_weights)
            output = tf.matmul(down_projected, output_weights) + output_bias
        else: #experiment A,B
            output = tf.matmul(state.h, output_weights) + output_bias
        output_list += [output]

    outputs = tf.stack(output_list, axis=1) # (batch_size x time_steps x embedding_size)

with tf.variable_scope("optimizer"):
    #calculate loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs[:,:-1,:], labels=targets[:,1:])) #TODO take sum instead of mean???
    #define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #clip gradients
    var = tf.trainable_variables()
    grad,_ = tf.clip_by_global_norm(tf.gradients(loss, var), 5)
    #training operation
    train_op = optimizer.apply_gradients(zip(grad,var))
    #logging
    tf.summary.scalar("loss", loss)
    log_op = tf.summary.merge_all()

#initialise graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if use_word2vec: #experiment B: load pretrained word2vec embedding weights
    load_embedding(sess, vocab, embedding_weights, "./wordembeddings-dim100.word2vec", embedding_size, vocab_size)

#logging
log_dir = "./log/"+datetime.datetime.now().strftime("%m%d_%H.%M")
writer = tf.summary.FileWriter(log_dir+"/summary/train", sess.graph)

#set up saving the  model
model_dir = os.path.abspath(os.path.join(log_dir, "checkpoints"))
model_prefix = os.path.join(model_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=number_checkpoint) #TODO does max to keep refer to first or last checkpoints?

#perform training step and return loss
def train(x, y, step):
    l,log,_ = sess.run([loss, log_op, train_op], feed_dict={inputs: x, targets: y}) #train
    writer.add_summary(log, step) #add (step,loss) to summary for visualising loss 
    return l

#DRIVER
batcher = Batcher(data)

#train
for step in range(n_train_steps):
    x = y = batcher.get_batch(batch_size)
    l = train(x, y, step)
    
    if step % 1 == 0:
        print("{:.1f}% loss={}".format(100*step/n_train_steps, l))
    if (step+1) % checkpoint_time == 0:
        model_path = saver.save(sess, model_prefix, global_step=step)
        
        print("Model saved to {}\n".format(model_path))

model_path = saver.save(sess, model_prefix, global_step=n_train_steps-1)
print("Final model saved to {}\n".format(model_path))

#logging
writer.close()

