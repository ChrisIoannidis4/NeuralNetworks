import tensorflow as tf
import matplotlib.pyplot as mpl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import timeit
(x_train, y_train), (x_test, y_test) =tf.keras.datasets.mnist.load_data()
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

x_train=x_train/255
x_test=x_test/255

knn=KNeighborsClassifier(n_neighbors=1)
t_start=timeit.default_timer()
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
t_stop=timeit.default_timer()

report=classification_report(y_test, y_pred)
print("classification report:\n", report)


acc=accuracy_score(y_test, y_pred)
print("accuracy: ", {acc:.4})
print("time: ", t_stop-t_start)