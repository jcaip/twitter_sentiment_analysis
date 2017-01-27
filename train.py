import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from text_rnn import TextRNN
import util
import time
import os


config = {
        'TRAINING_DATA_LOCATION': 'training.csv',
        'TESTING_DATA_LOCATION': 'testing.csv',
        'NUM_EXAMPLES': 10000,
        'BATCH_SIZE': 1000,
        'NUM_EPOCHS': 1,
        'LEARNING_RATE': 0.001,
        'HIDDEN_LAYER_SIZE': 150,
        'WORD_VECTOR_DIM': 200,
        'SAVE_LOCATION': os.path.join('./checkpoints/', str(time.time))
        }

print("# Loading training data")
training_data_raw = open(config['TRAINING_DATA_LOCATION'],'r',encoding='latin-1').readlines()
random.shuffle(training_data_raw)
num_examples = config['NUM_EXAMPLES']
training_data_raw= training_data_raw[:num_examples]

print("# Processing training data")
x_train, y_train, vocab_processor = util.load_training_data(training_data_raw)

print(" Loading and Processing testing data")
testing_data_raw = open(config['TESTING_DATA_LOCATION'],'r',encoding='latin-1').readlines()
x_test, y_test = util.load_testing_data(testing_data_raw, vocab_processor)

print("# Creating RNN")
rnn = TextRNN(x_train.shape[1], y_train.shape[1], config['HIDDEN_LAYER_SIZE'], 
        len(vocab_processor.vocabulary_), config['WORD_VECTOR_DIM'], l2_reg=0.0)
optimizer = tf.train.AdamOptimizer(config['LEARNING_RATE'])
minimizer = optimizer.minimize(rnn.loss)

print("# Initializing Tensorflow")
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
saver = tf.train.Saver()

print("# Training")
batch_size = config['BATCH_SIZE']
no_of_batches = int(len(training_data_raw)/batch_size)
epoch = config['NUM_EPOCHS']

losses = []
errors = []

for i in range(epoch):
    test_error = sess.run(rnn.error,
            {
                rnn.X: x_test,
                rnn.y: y_test,
                rnn.droput_keep_prob: 1.0
            })
    errors.append(float(test_error))

    ptr=0
    bar = tqdm(total=no_of_batches)
    for j in range(no_of_batches):
        inp, out = x_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
        status, loss = sess.run([minimizer, rnn.loss],
                {
                    rnn.X: inp,
                    rnn.y: out,
                    rnn.droput_keep_prob: 0.5
                })
        ptr += batch_size
        bar.set_description("Iteration: " + str(j) + " | Loss: " + str(loss) + " | Error: " + str(test_error))
        bar.update()
        losses.append(float(loss))

train_error, predictions= sess.run([rnn.error, rnn.prediction],
        {
            rnn.X: x_train[:batch_size],
            rnn.y: y_train[:batch_size],
            rnn.droput_keep_prob: 1.0
        })
print("Training Error: " + str(train_error))

pred, test_error = sess.run([rnn.prediction, rnn.error],
        {
            rnn.X: x_test,
            rnn.y: y_test,
            rnn.droput_keep_prob: 1.0
        })
print("Test Error: " + str(test_error))
print("Predictions: " + str(pred))
# print(y_test)


#plotting
plt.plot(range(len(losses)), losses)
plt.savefig('loss_function.png')
plt.clf()

plt.plot(range(len(errors)), errors)
plt.savefig('error.png')

saver.save(sess, config['SAVE_LOCATION'])
vocab_processor.save(config['SAVE_LOCATION'])
sess.close()
