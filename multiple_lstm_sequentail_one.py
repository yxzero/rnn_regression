#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: multiple_lstm_sequentail.py
#Author: yuxuan
#Created Time: 2016-05-11 20:05:19
############################
import pandas as pd
from random import random

flow = (list(range(1,10,1)) + list(range(10,1,-1)))*100
#pdata = pd.DataFrame({"a":flow, "b":flow})
pdata = pd.DataFrame({"a":flow})
#pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:] * random()  # some noise

import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Input, Embedding, LSTM, Dense, merge, Merge
from keras.layers.core import Reshape
from keras.models import Model
from keras.regularizers import l2, activity_l2
from keras.utils.visualize_util import plot

def _load_data(x_data, y_data, valid_portion=0.3):
    """
    data should be pd.DataFrame()
    """
    X_data_lstm = x_data[0] 
    X_data_c = x_data[1]
    X_data_scores = x_data[2]
    n_samples = len(X_data_lstm)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))

    test_set_x_lstm = [X_data_lstm[s] for s in sidx[n_train:]]
    test_set_x_c = [X_data_c[s] for s in sidx[n_train:]]
    test_set_x_s = [X_data_scores[s] for s in sidx[n_train:]]
    test_set_y = [[y_data[s]] for s in sidx[n_train:]]

    train_set_x_lstm = [X_data_lstm[s] for s in sidx[:n_train]]
    train_set_x_c = [X_data_c[s] for s in sidx[:n_train]]
    train_set_x_s = [X_data_scores[s] for s in sidx[:n_train]]
    train_set_y = [[y_data[s]] for s in sidx[:n_train]]
    print(train_set_x_s)
    exit()

    train_x = [
                np.array(train_set_x_lstm),np.array(train_set_x_s),np.array(train_set_x_c),
            ]
    train_y = np.array(train_set_y)
    test_x = [
                np.array(test_set_x_lstm),np.array(test_set_x_s),np.array(test_set_x_c),
            ]
    test_y = np.array(test_set_y)
    return (train_x, train_y),(test_x, test_y)

class MultipleLSTM():
    def __init__(self, input_dim=4, batch_size=700, lstm_hidden=6, lstm_timesteps=12, output_dim=1):
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        self.model = None
        self.batch_size = batch_size 
        self.lstm_timesteps = lstm_timesteps

    def build_model(self):
        lstm_branch = []
        input_branch = []
        main_input = Input(shape=(self.lstm_timesteps, self.input_dim),
                name='main_input')
        input_branch.append(main_input)
        lstm_out = LSTM(self.lstm_hidden, return_sequences=True)(main_input)
        auxiliary_input = Input(shape=(1,self.lstm_timesteps), name='auxiliary_input')
        input_branch.append(auxiliary_input)
        x1 = merge([auxiliary_input, lstm_out], mode='dot', dot_axes=[2,1], name='merge_lstm_auxi')
        flatten = Reshape((self.lstm_hidden,))(x1)
        c_input = Input(shape=(6,), name='c_input')
        input_branch.append(c_input)
        x2 = merge([flatten, c_input], mode='concat') 
        lstm_out = Dense(self.lstm_hidden, activation='relu', W_regularizer=l2(0.001),
            activity_regularizer=activity_l2(0.001), name="lstm_out")(x2)
        '''
        lstm_out_1 = Dense(self.lstm_hidden, activation='relu', W_regularizer=l2(0.001),
            activity_regularizer=activity_l2(0.001), name="lstm_out_1")(lstm_out)
        '''
        final_loss = Dense(self.output_dim, name='main_output')(lstm_out)
        self.model = Model(input_branch, output=final_loss)
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop")
        plot(self.model, to_file='multiple_model_one.png', show_shapes=True)
    
    def train_model(self, X_train, Y_train, nb_epoch=3000, validation_split=0.05):
        self.model.fit(X_train, Y_train, batch_size=self.batch_size,
                nb_epoch=nb_epoch, validation_split=validation_split)
        '''
        for layer_i in self.model.layers:
            print(layer_i.get_config())
            print(layer_i.get_weights())
        '''

    def prediction(self, X_test, Y_test):
        import matplotlib.pyplot as plt
        target = self.model.predict(X_test)
        #10 **
        #target = 10 ** target
        #Y_test = 10 ** Y_test
        test_error = np.sqrt(((target - Y_test) ** 2).mean(axis=0))
        print test_error
        plt.xlabel("target")
        plt.ylabel("predict")
        plt.plot(Y_test, target, '.')
        plt.plot([0,80000],[0,80000])
        plt.show()
        return target, test_error

def Load_data():
    xfile = open("X1.data", 'r')
    x_str = xfile.read()
    xfile.close()
    yfile = open("Y1.data", 'r')
    y_str = yfile.read()
    yfile.close()
    x_data = eval(x_str)
    y_data = eval(y_str)
    return _load_data(x_data, y_data)

if __name__ == "__main__":
    (train_x, train_y),(test_x, test_y) = Load_data()
    mp_lstm = MultipleLSTM()
    mp_lstm.build_model()
    mp_lstm.train_model(train_x, train_y)
    target, testerror = mp_lstm.prediction(test_x, test_y)
    result = open("result.csv",'w')
    result.write('1,1.5,2,2.5,预测,真实\n')
    for i in range(len(target)):
        result.write(str(test_x[2][i][0])+','+str(test_x[2][i][1])+\
                ','+str(test_x[2][i][2])+','+str(test_x[2][i][3])+',')
        result.write(str(target[i][0])+','+str(test_y[i][0])+'\n')
    result.close()
