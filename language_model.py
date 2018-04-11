import tensorflow as tf
import numpy as np

class NeuralLanguageModel:
    def __init__(self, sentence_len=30, vocab_size=100):
        self.hidden_size = 32
        self.batch_size = 1
        self.learning_rate = 0.001
        self.input_n = sentence_len
        self.output_n = vocab_size
        #TODO missing vocab information?

        #construct neural network
        tf.reset_default_graph()
        
        rnn = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden_size)

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_n], name="inputs") #TODO float32->int32?
        initial_state = state = tf.zeros([self.batch_size, self.hidden_size])

        #input_embeddings tf.nn.embedding_lookup(embedding_matrix, inputs_ids) TODO

        for i in range(self.input_n):
            output, state = rnn(tf.reshape(self.inputs[:,i], [self.batch_size,1]), state)
        
        self.final_state = state 
        self.softmax = tf.nn.softmax(self.final_state)
        self.prediction = tf.argmax(self.softmax, axis=1)
        
        #TODO de-embed prediction
        
        #TODO target, loss, optimizer

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, x, y):
        #loss,_ = self.sess.run([loss, optimiser], feed_dict={self.inputs: x, self.targets: y}) TODO
        return loss

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.inputs: x})[0] #TODO prediction for every timestep, not just final?

data = [i for i in range(30)]
nn = NeuralLanguageModel()
print(nn.predict(np.reshape(data, [1,30])))
