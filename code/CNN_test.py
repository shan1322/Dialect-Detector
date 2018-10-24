import numpy as np
from keras.models import load_model

x_train = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test = np.load('../MAT/test_y.npy')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = load_model("../Models/CNN.h5")
pred = model.evaluate(x=x_test, y=y_test, verbose=2)
print("accuracy", pred[1])
print("loss", pred[0])
