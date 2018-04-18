import tensorflow as tf
import numpy as np
from preprocess import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='checkpoint_path', type=str, default="./log/15:17:06.705800/checkpoints/model-10880")
parser.add_argument('--sample', dest='sample', action='store_true')
args = parser.parse_args()
checkpoint_path = args.checkpoint_path
sample = args.sample

n = 20 #generated sentence max length

vocab,inv_vocab = load_vocab()
data = load_continuation_data() #TODO should continuation data contain string words rather than int code words?

hidden_size = 512
vocab_size = len(vocab)
embedding_size = 100

#construct neural network graph
embedding_weights = tf.get_variable("embeddings/embedding_weights", shape=[vocab_size, embedding_size])
output_weights = tf.get_variable("rnn/output_weights", shape=[hidden_size, vocab_size])
output_bias = tf.get_variable("rnn/output_bias", shape=[vocab_size])
rnn = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, name="rnn/lstm_cell")
input_word = tf.placeholder(shape=[], dtype=tf.int32, name="input_word")
input_word_embedded = tf.reshape(tf.nn.embedding_lookup(embedding_weights, input_word), [1,-1])
input_state = tf.placeholder(shape=[1,hidden_size], dtype=tf.float32, name="input_state")
input_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(input_state, input_state) #tuple necessary for TF LSTM
_, output_state = rnn(input_word_embedded, input_state_tuple)
output = tf.matmul(output_state.h, output_weights) + output_bias
output_softmax = tf.nn.softmax(output)
prediction = tf.argmax(output_softmax,axis=1) #TODO softmax unnecessary if not sampling?

sess = tf.Session()

#load weights
saver = tf.train.Saver(var_list=[embedding_weights, output_weights, output_bias, rnn.trainable_variables[0], rnn.trainable_variables[1]])
saver.restore(sess, checkpoint_path) #TODO automatically locate

def predict(input_word_, input_state_, sample=False):
    if not sample:
        #word based on argmax (highest probability word)
        predict,out_state = sess.run([prediction,output_state.h], feed_dict={input_word: input_word_, input_state: np.reshape(input_state_,[1,-1])})
        predict = predict[0]
    else:
        #sample words based on softmax distribution
        softmax,out_state = sess.run([output_softmax,output_state.h], feed_dict={input_word: input_word_, input_state: np.reshape(input_state_,[1,-1])})
        predict = np.random.choice(range(vocab_size), p=softmax[0]) #random vocabulary index, weighted by softmax probabilities 
    return predict,out_state

if not os.path.exists("./results"):
  os.makedirs("./results")
write_file = open("./results/group08.continuation", 'w')

count = 0 #debug
output_sentences = []
for sentence in data:
    #generate new words after sentence until <eos> is generated or sentence length reaches n
    i = 0 #current number of words in sentence
    output_sentence = []
    state = np.zeros([hidden_size]) #initial state
    _, state = predict(vocab['<bos>'], state) #assume initial <bos> token
    #words in sentence
    for word in sentence:
        predicted_word, state = predict(word, state)
        output_sentence += [inv_vocab[word]]
        i += 1
    #generate new words
    stop_word_predicts = {'<unk>': 0, '<bos>': 0, '<pad>': 0} #debug
    while i < n and predicted_word != vocab['<eos>']:
        new_predicted_word, new_state = predict(predicted_word, state, sample=sample)
        if sample:
            #only accept word if it is not stop word
            if new_predicted_word not in [vocab[stop_word] for stop_word in ['<unk>', '<bos>', '<pad>']]:
                predicted_word = new_predicted_word
                state = new_state
                output_sentence += [inv_vocab[predicted_word]]
                i += 1
            #else: #debug
                #stop_word_predicts[inv_vocab[new_predicted_word]] += 1 #debug
        else:
            predicted_word = new_predicted_word
            state = new_state
            output_sentence += [inv_vocab[predicted_word]]
            i += 1

    #write sentence to file
    write_str = ' '.join(str(word) for word in output_sentence)
    write_str += '\n'
    write_file.write(write_str)
    
    count += 1 #debug
    if count % 20 == 0: #debug
        print_str = "" #debug
        for w in output_sentence: #debug
            print_str += (w + " ") #debug        
        print(print_str) #+ "(bad predictions=" + str(stop_word_predicts) + ")") #debug
