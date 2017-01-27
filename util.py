import re
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import learn

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def process_output(x):
    if x == '0':
        return [1, 0]
    elif x == '4':
        return [0, 1]
    else:
        return [0, 0]

def load_training_data(training_data_raw):
    train_input = []
    train_output = []

    for example in tqdm(training_data_raw):
        example = example.replace('"','')
        temp = example.split(',')
        if temp[0] =='0' or temp[0] =='4':
            train_output.append(process_output(temp[0]))
            train_input.append(clean_str(temp[5]))

    max_sentence_length = max(len(x.split(' ')) for x in train_input)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    return (np.array(list(vocab_processor.fit_transform(train_input))), np.array(train_output), vocab_processor)

def load_testing_data(testing_data_raw, vocab_processor):
    test_input = []
    test_output = []
    for example in tqdm(testing_data_raw):
        example = example.replace('"','')
        temp = example.split(',')
        if temp[0] =='0' or temp[0] =='4':
            test_output.append(process_output(temp[0]))
            test_input.append(temp[5])

    x_test = np.array(list(vocab_processor.fit_transform(test_input)))
    y_test = np.array(test_output)
    return(x_test, y_test)
