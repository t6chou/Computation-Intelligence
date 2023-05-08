from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
import random


def f1(x):
    return x*np.sin(6*np.pi*x)*np.exp(-(x^2))

def f2(x):
    return np.exp(-x^2)*np.arctan(x)*np.sin(4*np.pi*x)

def generate(n):
    # generate dataset
    x_train = np.random.sample(range(-1, 1), 10)
    y_train = np.empty(3)
    for i in range(len(x_train)):
        y_train[i] = f1(x_train[i])
        
        
    
def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # explore the dataset
    print('Training set shape: {}'.format(x_train.shape))
    print('Training labels shape: {}'.format(y_train.shape))
    print('Test set shape: {}'.format(x_test.shape))
    print('Test labels shape: {}'.format(y_test.shape))
    
    
    
if __name__ == "__main__":
    main()