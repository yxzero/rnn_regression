import pandas as pd
from random import random

flow = (list(range(1,10,1)) + list(range(10,1,-1)))*100
pdata = pd.DataFrame({"a":flow, "b":flow})
#pdata = pd.DataFrame({"a":flow})
pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:] * random()  # some noise

import numpy as np
from matplotlib import pyplot as plt

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

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 2
hidden_neurons = 50

model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data

print(X_train.shape)
print(y_train.shape)
'''
tempx = X_train[0:2,0:4,:]
print tempx
tempx = np.transpose(tempx, (1,0,2))
print(tempx.shape)
print(tempx)
print y_test
'''
model.fit(X_train, y_train, batch_size=700, nb_epoch=50, validation_split=0.05)
predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
#print(rmse)
# and maybe plot it
#pd.DataFrame(predicted).to_csv("predicted.csv")
#pd.DataFrame(y_test).to_csv("test_data.csv")

plt.plot(range(0,50), predicted[0:50,0])
plt.plot(range(0,50), y_test[0:50,0])
#plt.plot(range(0,50), predicted[0:50,1])
#plt.plot(range(0,50), y_test[0:50,1])
plt.show()

