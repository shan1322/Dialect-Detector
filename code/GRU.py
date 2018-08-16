import numpy as np
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM,GRU
from sklearn.metrics import classification_report
from keras.models import model_from_json
from keras.models import load_model
# Load numpy files for train and test set
x_train = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test= np.load('../MAT/train_y.npy')
x_val=np.load('../MAT/val_x.npy')
y_val=np.load('../MAT/val_y.npy')

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)
# build gru model
def GRUmodel(x_train):
    model = Sequential()
    model.add(GRU(64, activation='tanh',return_sequences=True, stateful=False,batch_input_shape= (batch_size, x_train.shape[1],x_train.shape[2] )))   #  timestep,datadim
    model.add(GRU(64, activation='tanh',return_sequences=True, stateful=False))
    model.add(GRU(64,activation='tanh', stateful=False))
    model.add(Dropout(.25))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    return model
# define paramaetes
batch_size = 25
nb_classes = 214
# train step
model=GRUmodel(x_train)
model.fit(x_train, y_train, batch_size=batch_size, epochs=1000,validation_data=(x_val, y_val))
# save model
model.save('../Models/GRU.h5')
