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
from keras.model import Sequence
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Merge
from keras.regularizers import l2, activity_l2

def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))
    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

class MultipleLSTM():
    def __init__(input_dim=10, layers_number=5, lstm_hidden=50, output_dim=1):
        self.input_dim = input_dim
        self.layers_number = layers_number
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        self.model = None

    def build_model(self):
        lstm_branch = []
        for i in range(layers_number):
            model = Sequential()
            model.add(LSTM(self.lstm_hidden, input_dim=self.input_dim,
                return_sequences=False))
            lstm_branch.append(model)
        merged = Merge(lstm_branch)
        final_model = Sequence()
        final_model.add(merged)
        final_model.add(Dense(self.output_dim, input_dim=self.lstm_hidden,
            W_regularizer=l2(0.001), activity_regularizer=activity_l2(0.001)))
        final_model.add(Activation('linear'))
        self.model = final_model
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop")
    
    def train_model(self, X_train, Y_train, batch_size=700, nb_epoch=50, validation_split=0.05):
        self.model.fit(X_train, Y_train, batch_size=batch_size,
                nb_epoch=nb_epoch, validation_split=validation_split)

    def prediction(self, X_test, Y_test):
        target = self.model.predict(X_test)
        test_error = ((target - Y_test) ** 2).means(axis=0)
        return target, test_error

class Load_data():
    def __init__():
        pass
