import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential

x_train = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test = np.load('../MAT/train_y.npy')
y_val = np.load('../MAT/val_y.npy')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

batch_size = 25
nb_classes = 214


def modelDense(x_train):
    model = Sequential()
    model.add(Dense(64, activation='sigmoid', input_shape=(x_train.shape[1],)))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(1024, activation='sigmoid'))
    model.add(Dense(2048, activation='sigmoid'))
    model.add(Dense(2048, activation='sigmoid'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = modelDense(x_train)

model.fit(x_train, y_train, batch_size=batch_size, epochs=100)
