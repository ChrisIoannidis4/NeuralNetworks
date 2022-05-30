from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import timeit
from keras.datasets import cifar10

(x_train, y_train) , (x_test, y_test)=cifar10.load_data()

x_train=x_train.reshape(50000,3072)
x_test=x_test.reshape(10000,3072)

x_train=x_train[:10000, :]
x_test=x_test[:2500, :]
y_train=y_train[:10000]
y_test=y_test[:2500]

x_train=x_train/255
x_test=x_test/255


t_start=timeit.default_timer()
model= NearestCentroid(metric='euclidean')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
t_stop=timeit.default_timer()
print("Classification report:\n", classification_report(y_test,y_pred))
print("accuracy: ", accuracy_score(y_test, y_pred))
print("time: ", t_stop-t_start)