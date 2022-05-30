import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



#loading the mnist handwritten digits dataset, 28x28 pixel images
(x_train, y_train), (x_test, y_test)= mnist.load_data()

#we normalize the data for computational conveniece
x_train=x_train/255.0
x_test=x_test/255.0
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
'''
#PCA
scaler = StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test1=scaler.transform(x_test)
pca = PCA(.75)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test1 = pca.transform(x_test)

newx_test=pca.inverse_transform(x_test1)

import cv2 as cv
import numpy as np
img=cv.imread(f'digit.png')[:,:,0]
img=np.invert(np.array([img]))
img=img/255
img=img.reshape(1, 784)
img1=pca.transform(img)
pcaim=pca.inverse_transform(img1)
plt.imshow(pcaim.reshape(28,28), cmap="gray")
plt.show()

ten=10
for i in range(ten):

 fig=plt.subplot(2,1,1)
 plt.imshow(x_test[i].reshape(28,28), cmap="gray")
 fig=plt.subplot(2,1,2)
 plt.imshow(newx_test[i].reshape(28,28), cmap='gray')
 plt.show()



'''

autoencoder=tf.keras.models.Sequential()

#encoding phase
autoencoder.add(tf.keras.layers.Dense(128, activation="relu"))

#bottleneck
autoencoder.add(tf.keras.layers.Dense(32, activation="relu"))

#decoding phase
autoencoder.add(tf.keras.layers.Dense(128, activation="relu"))
autoencoder.add(tf.keras.layers.Dense(331, activation="sigmoid"))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256)




out_images=autoencoder.predict(x_test, batch_size=256)


out_images=out_images.reshape(10000,28,28)
plt.figure(figsize=(20,4))
x_test=x_test.reshape(10000,28,28)



ten=10
for i in range(ten):

 fig=plt.subplot(2,1,1)
 plt.imshow(x_test[i], cmap="gray")
 fig=plt.subplot(2,1,2)
 plt.imshow(out_images[i], cmap='gray')
 plt.show()


import cv2 as cv
import numpy as np
img=cv.imread(f'digit.png')[:,:,0]
img=np.invert(np.array([img]))
img=img/255
img=img.reshape(1, 784)
newimg=autoencoder.predict(img)
img=img.reshape(28, 28)
newimg=newimg.reshape(28, 28)
fig=plt.subplot(2,1,1)
plt.imshow(img, cmap='gray')
fig=plt.subplot(2,1,2)
plt.imshow(newimg, cmap='gray')

plt.show()
