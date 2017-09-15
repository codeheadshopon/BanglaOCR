# coding=utf8
'''This example uses a convolutional stack followed by a recurrent stack
and a CTC logloss function to perform optical character recognition
of generated text images. I have no evidence of whether it actually
learns general shapes of text, or just is able to recognize all
the different fonts thrown at it...the purpose is more to demonstrate CTC
inside of Keras.  Note that the font list may need to be updated
for the particular OS in use.
This starts off with 4 letter words.  For the first 12 epochs, the
difficulty is gradually increased using the TextImageGenerator class
which is both a generator class for test/train data and a Keras
callback class. After 20 epochs, longer sequences are thrown at it
by recompiling the model to handle a wider image and rebuilding
the word list to include two words separated by a space.
The table below shows normalized edit distance values. Theano uses
a slightly different CTC implementation, hence the different results.
            Norm. ED
Epoch |   TF   |   TH
------------------------
    10   0.027   0.064
    15   0.038   0.035
    20   0.043   0.045
    25   0.014   0.019
This requires cairo and editdistance packages:
pip install cairocffi
pip install editdistance
Created by Mike Henry
https://github.com/mbhenry/
'''
import os
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import cPickle,gzip
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,Flatten,Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image

import matplotlib.pyplot as plt
import keras.callbacks
import numpy as np
from keras.models import Sequential
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

OUTPUT_DIR = 'image_ocr'

np.random.seed(55)


bind = {}
C = 0
banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়0"
for i in banglachars.decode('utf-8'):
        bind[C]=i
        # print(C)
        C+=1


def text_to_labels(text,num_classes):
        ret=[]
        print(text)
        for l in text.decode('utf-8'):
                for i in range(0, 47):
                        if (bind[i] == l):
                                ret.append(i)


        return ret



# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    # print(out)
    ret = []
    # print(out.shape[0])
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        # print(out_best)
        for c in out_best:
            print(c)
            outstr+=bind[c]

        print(outstr)   
    return ret

class OwnVizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            # word_batch = next(self.text_img_gen)[0]
            word_batch=self.text_img_gen
            print(word_batch.shape[0])
            num_proc = min(word_batch.shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(0, num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
       
        # self.show_edit_distance(256)
        # word_batch = next(self.text_img_gen)[0]
        word_batch=self.text_img_gen
        print("WordBatch",word_batch['the_input'].shape[0])

        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            # pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()


def train(run_name, start_epoch, stop_epoch, img_w):
    # Input Parameters
    img_h = 64
    words_per_epoch = 300
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 256
    minibatch_size = 32

    ''' Spatial Transformer Layers'''
    # initial weights
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]



    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
   

    act = 'relu'
    
    downsample = 1.
    w = np.zeros((20, 6))
    b = np.zeros((6,))

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    # input_data = (SpatialTransformer(localization_net=locnet,
    #                          output_size=(img_w,img_h), input_shape=input_shape))(input_data)

    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    # print("Output Size",img_gen.get_output_size())

    inner = Dense(47, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    # Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[10], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)



    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    test_func = K.function([input_data], [y_pred])
       
    (X_train, y_train, train_input_length, train_labels_length,train_source_str), (X_test, y_test, test_input_length, test_labels_length,test_source_str) = dataset_load('./OCR_BanglaData.pkl.gz')


    X_train = X_train.reshape(X_train.shape[0], 128,64,1)
    X_test = X_test.reshape(X_test.shape[0], 128,64,1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

 
    outputs_1 = {'ctc': np.zeros([40000])}
    outputs_2 = {'ctc': np.zeros([10000])}
    inputs_1 = {'the_input': X_train,
                  'the_labels': y_train,
                  'input_length': train_labels_length,
                  'label_length': train_input_length,
                  'source_str':train_source_str
                  }
    inputs_2 = {'the_input': X_test,
              'the_labels': y_test,
              'input_length': test_labels_length,
              'label_length': test_input_length,
              'source_str':test_source_str
              }
    viz_cb = OwnVizCallback(run_name, test_func, image_gen)
   
    for i in range(1000,1010):
        print(train_labels_length[i])
        print(y_train[i])
        print(X_train[i])
        print(train_input_length)
        
    # model.fit([np.array(X_train),np.array(y_train),np.array(train_input_length),np.array(train_labels_length)],
    #     outputs_1, batch_size=28, epochs=120, verbose=1,callbacks=[viz_cb],
    #     validation_data=([np.array(X_test),np.array(y_test),np.array(test_input_length),np.array(test_labels_length)]
    #         ,outputs_2))


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    train(run_name, 0, 20, 128)
    # increase to wider images and start at epoch 20. The learned weights are reloaded
train(run_name, 20, 25, 512)