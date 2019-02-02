import numpy as np
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# Load numpy files for train and test set
x_train_data = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test = np.load('../MAT/test_y.npy')

print(x_train_data.shape)
print(x_test.shape)


# build gru model
def gru_model(x_train):
    model = Sequential()
    model.add(LSTM(1024, batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1000))
    model.add(Dense(214, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
    return model


# define paramaetes
nb_classes = 214
n_batch = 1709
# train step
model = gru_model(x_train_data)
model.fit(x_train_data, y_train, epochs=50, batch_size=n_batch, verbose=1)
# save model
model.save('../Models/GRU.h5')

demo_model = Sequential()
n_batch = 1
demo_model.add(LSTM(1024, batch_input_shape=(n_batch, x_train_data.shape[1], x_train_data.shape[2])))
demo_model.add(Dense(1000))
demo_model.add(Dense(214, activation='softmax'))
# copy weights
old_weights = model.get_weights()
demo_model.set_weights(old_weights)
demo_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
demo_model.save('../Models/LSTM.h5')

test_model = Sequential()
n_batch = 428
test_model.add(LSTM(1024, batch_input_shape=(n_batch, x_train_data.shape[1], x_train_data.shape[2])))
test_model.add(Dense(1000))
test_model.add(Dense(214, activation='softmax'))
# copy weights
old_weights = model.get_weights()
test_model.set_weights(old_weights)
test_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
test_model.save('../Models/LSTM1.h5')
