import numpy as np
from keras.layers.core import Dense,Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential

# Load numpy files for train and test set
x_train = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test = np.load('../MAT/test_y.npy')
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



# build gru model
def GRUmodel(x_train):
    model = Sequential()
    model.add(GRU(64, activation='tanh', return_sequences=True, stateful=False,batch_input_shape=(x_train.shape[0], x_train.shape[1], x_train.shape[2])))  # timestep,datadim
    model.add(GRU(128, activation='tanh', return_sequences=True, stateful=False))
    model.add(GRU(128, activation='tanh', return_sequences=True, stateful=False))
    model.add(GRU(128, activation='tanh', return_sequences=True, stateful=False))
    model.add(GRU(128, activation='tanh', return_sequences=True, stateful=False))
    model.add(GRU(128, activation='tanh', return_sequences=True, stateful=False))
    model.add(GRU(256, activation='tanh', return_sequences=True, stateful=False))
    model.add(GRU(256, activation='tanh', stateful=False))
    model.add(Dropout(.25))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


# define paramaetes
nb_classes = 214
# train step
model = GRUmodel(x_train)
model.fit(x_train, y_train,batch_size=x_train.shape[0], epochs=200)
# save model
model.save('../Models/GRU.h5')
