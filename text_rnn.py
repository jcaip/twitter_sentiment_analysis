import tensorflow as tf
import numpy as np

class TextRNN(object):

    def __init__(self, input_layer_size, output_layer_size, hidden_layer_size, vocab_size, embedding_size, l2_reg=0.0):
        self.X = tf.placeholder(tf.int32, [None, input_layer_size], name='X')
        self.y = tf.placeholder(tf.float32, [None, output_layer_size], name='y')
        self.droput_keep_prob = tf.placeholder(tf.float32, name="droput_keep_prob")

        #training word embeddings
        with tf.device('/cpu:0'), tf.name_scope('embeddings'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0,1.0))
            self.embedding = tf.nn.embedding_lookup(W, self.X)

        #creating the rnn
        network = tf.nn.rnn_cell.LSTMCell(hidden_layer_size, state_is_tuple=True)
        #applying droput
        network = tf.nn.rnn_cell.DropoutWrapper(network, output_keep_prob=self.droput_keep_prob)

        #multiple layers
        # network = tf.nn.rnn_cell.MultiRNNCell([network] * 4)

        val, state = tf.nn.dynamic_rnn(network, self.embedding, dtype=tf.float32)

        val = tf.transpose(val, [1,0,2])
        last = tf.gather(val, int(val.get_shape()[0])-1)

        W = tf.Variable(tf.truncated_normal([hidden_layer_size, output_layer_size]), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[output_layer_size]), name='b')

        with tf.name_scope('output'):
            self.scores= tf.matmul(last, W) + b
            self.prediction = tf.argmax(self.scores, 1)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y))

        with tf.name_scope("error"):
            mistakes = tf.not_equal(tf.argmax(self.y,1), tf.argmax(self.scores, 1))
            self.error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

