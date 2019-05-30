# Created by YUAN at 2019-05-03
"""
Feature: #Enter feature name here
# Enter feature description here
Scenario: #Enter scenario name here
# Enter steps here
Test File Location: # Enter]
"""

# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import newaxis

warnings.filterwarnings("ignore")


def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')
    print(type(data), type(data[0]), data)

    print('data len:', len(data))
    print('sequence len:', seq_len)

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length]) 

    print('result len:', len(result))
    print('result shape:', np.array(result).shape)
    print(result[:1])

    if normalise_window:
        result = normalise_windows(result)

    print(result[:1])
    print('normalise_windows result shape:', np.array(result).shape)

    result = np.array(result)

    # 划分train、test
    row = round(0.9 * result.shape[0])
    train = result[:row, :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data: 
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def build_model(layers):  
    model = Sequential()

    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    # model.compile(loss="mse", optimizer="rmsprop")
    model.compile(loss="mae", optimizer="rmsprop")
    # model.compile(loss="mse", optimizer="sgd")
    # model.compile(loss="mse", optimizer="adagrad")
    # model.compile(loss="mse", optimizer="adadelta")
    # model.compile(loss="mse", optimizer="adam")
    print("Compilation Time : ", time.time() - start)
    return model


# 直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:', np.array(predicted).shape) 
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    plt.savefig(filename + '.png')




if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 1
    seq_len = 50

    print('> Loading data... ')

    X_train, y_train, X_test, y_test = load_data('UKUSD.csv', seq_len, True)
    #X_train, y_train, X_test, y_test = load_data('EURUSD.csv', seq_len, True)
    #X_train, y_train, X_test, y_test = load_data('AUSUSD.csv', seq_len, True)

    print('X_train shape:', X_train.shape) 
    print('y_train shape:', y_train.shape)  
    print('X_test shape:', X_test.shape) 
    print('y_test shape:', y_test.shape) 

    print('> Data Loaded. Compiling...')

    model = build_model([1, 50, 100, 1])

    model.fit(X_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.05)
    model.save('./lstm_exchange.model')
   

    point_by_point_predictions = predict_point_by_point(model, X_test)
    print('point_by_point_predictions shape:', np.array(point_by_point_predictions).shape)  

    print('Training duration (s) : ', time.time() - global_start_time)

    plot_results(point_by_point_predictions, y_test, 'point_by_point_predictions')
