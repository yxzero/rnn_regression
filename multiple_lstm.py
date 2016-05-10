#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: rnn_to_regression.py
#Author: yuxuan
#Created Time: 2016-04-24 19:00:28
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
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Merge
from keras.regularizers import l2, activity_l2
from keras.utils.visualize_util import plot

def _load_data(x_data, y_data, valid_portion=0.3):
    """
    data should be pd.DataFrame()
    """
    X_data_lstm1 = x_data[0]
    X_data_lstm2 = x_data[1]
    X_data_lstm3 = x_data[2]
    X_data_c1 = x_data[3]
    X_data_c2 = x_data[4]
    X_data_c3 = x_data[5]
    n_samples = len(X_data_lstm1)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))

    test_set_x_lstm1 = [X_data_lstm1[s] for s in sidx[n_train:]]
    test_set_x_lstm2 = [X_data_lstm2[s] for s in sidx[n_train:]]
    test_set_x_lstm3 = [X_data_lstm3[s] for s in sidx[n_train:]]
    test_set_x_c1 = [X_data_c1[s] for s in sidx[n_train:]]
    test_set_x_c2 = [X_data_c2[s] for s in sidx[n_train:]]
    test_set_x_c3 = [X_data_c3[s] for s in sidx[n_train:]]
    test_set_y = [[y_data[s]] for s in sidx[n_train:]]

    train_set_x_lstm1 = [X_data_lstm1[s] for s in sidx[:n_train]]
    train_set_x_lstm2 = [X_data_lstm2[s] for s in sidx[:n_train]]
    train_set_x_lstm3 = [X_data_lstm3[s] for s in sidx[:n_train]]
    train_set_x_c1 = [X_data_c1[s] for s in sidx[:n_train]]
    train_set_x_c2 = [X_data_c2[s] for s in sidx[:n_train]]
    train_set_x_c3 = [X_data_c3[s] for s in sidx[:n_train]]
    train_set_y = [[y_data[s]] for s in sidx[:n_train]]

    train_x = [
                np.array(train_set_x_lstm1),np.array(train_set_x_c1),
                np.array(train_set_x_lstm2),np.array(train_set_x_c2),
                np.array(train_set_x_lstm3),np.array(train_set_x_c3),
            ]
    train_y = np.array(train_set_y)
    test_x = [
                np.array(test_set_x_lstm1),np.array(test_set_x_c1),
                np.array(test_set_x_lstm2),np.array(test_set_x_c2),
                np.array(test_set_x_lstm3),np.array(test_set_x_c3),
            ]
    test_y = np.array(test_set_y)
    return (train_x, train_y),(test_x, test_y)

class MultipleLSTM():
    def __init__(self, input_dim=6, layers_number=3, lstm_hidden=25, lstm_timesteps=5, output_dim=1):
        self.input_dim = input_dim
        self.layers_number = layers_number
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        self.model = None
        self.lstm_timesteps = lstm_timesteps

    def build_model(self):
        lstm_branch = []
        for i in range(self.layers_number):
            model1 = Sequential()
            #model1.add(LSTM(self.lstm_hidden, input_dim=self.input_dim, return_sequences=False)) 
            model1.add(LSTM(self.lstm_hidden,
                input_shape=(self.lstm_timesteps, self.input_dim), return_sequences=False)) 
            model2 = Sequential()
            model2.add(Dense(output_dim=4, input_dim=4, activation='relu', W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001)))
            #model2.add(Dense(output_dim=4, input_dim=4))
            model = Sequential()
            model.add(Merge([model1,model2], mode='concat'))
            #model.add(Dense(self.lstm_hidden, activation='relu', W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001)))
            lstm_branch.append(model)
        merged = Merge(lstm_branch, mode='concat')
        final_model = Sequential()
        final_model.add(merged)
        final_model.add(Dense(self.output_dim, input_dim=self.lstm_hidden,
            W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001)))
        final_model.add(Activation('linear'))
        self.model = final_model
        self.model.compile(loss="mean_squared_error", optimizer="adadelta")
        plot(self.model, to_file='model.png', show_shapes=True)
    
    def train_model(self, X_train, Y_train, batch_size=700, nb_epoch=5000, validation_split=0.05):
        self.model.fit(X_train, Y_train, batch_size=batch_size,
                nb_epoch=nb_epoch, validation_split=validation_split)

    def prediction(self, X_test, Y_test):
        import matplotlib.pyplot as plt
        target = self.model.predict(X_test)
        test_error = np.sqrt(((target - Y_test) ** 2).mean(axis=0))
        print test_error
        plt.plot(Y_test, target, '.')
        plt.show()
        return target, test_error

def Load_data():
    xfile = open("X.data", 'r')
    x_str = xfile.read()
    xfile.close()
    yfile = open("Y.data", 'r')
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
    mp_lstm.prediction(test_x, test_y)

