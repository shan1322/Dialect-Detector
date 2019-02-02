import numpy as np
from keras.models import load_model

x_train = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test = np.load('../MAT/test_y.npy')
model = load_model("../Models/LSTM1.h5")
pred = model.evaluate(x=x_test, y=y_test, verbose=1,batch_size=855)
print(pred)
