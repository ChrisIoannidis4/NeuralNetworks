from sys import platform
import tensorflow as tf
import matplotlib.pyplot as mplt
import numpy as np
import pandas as pd
import timeit
#Load the data, and convert them from 3D to 2D arrays 
(x_train, y_train), (x_test, y_test) =tf.keras.datasets.mnist.load_data()


x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
#Normalization
x_train=x_train/255
x_test=x_test/255

mlpmodel=tf.keras.models.Sequential()
mlpmodel.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
mlpmodel.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

mlpmodel.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
t_start=timeit.default_timer()
mlpmodel.fit(x_train, y_train, epochs=15)

val_loss, val_acc=mlpmodel.evaluate(x_test,y_test)
t_stop=timeit.default_timer()
print("loss: ", val_loss, " accuracy: ", val_acc)
print("time: ", t_stop-t_start)