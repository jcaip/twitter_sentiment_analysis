import tensorflow as tf
from text_rnn import TextRNN

sess = tf.Session()
new_saver = tf.train.import_meta_graph('./10000examples_lr0.001_epochs100/rnn.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./10000examples_lr0.001_epochs100/'))
all_vars = tf.trainable_variables()
for v in all_vars:
    print(v)

rnn = TextRNN(x_train.shape[1], y_train.shape[1], 100, len(vocab_processor.vocabulary_), 200, l2_reg=0.0)
rnn = TextRnn
