from sklearn import svm
from keras.datasets import cifar10
from keras.utils import np_utils
from sklearn import metrics
import timeit

(x_train, y_train) , (x_test, y_test)=cifar10.load_data()

x_train=x_train.reshape(50000,3072)
x_test=x_test.reshape(10000,3072)

x_train=x_train[:10000, :]
x_test=x_test[:2500, :]
y_train=y_train[:10000]
y_test=y_test[:2500]

x_train=x_train/255
x_test=x_test/255

cls=svm.SVC(kernel="linear", gamma=1, C=1)
t_start=timeit.default_timer()
cls.fit(x_train,y_train)
pred=cls.predict(x_test)
t_stop=timeit.default_timer()
print("accuracy:  ", metrics.accuracy_score(y_test,pred))
print("precision: ", metrics.precision_score(y_test,pred, average= None))
print("recall:  ", metrics.recall_score(y_test, pred, average=None))
print(metrics.classification_report(y_test,pred))
print("time: ", t_stop-t_start)
