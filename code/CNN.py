import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten
from keras.layers.core import Dense
from keras.models import Model
from keras.callbacks import History

x_train = np.load('../MAT/train_x.npy')
x_test = np.load('../MAT/test_x.npy')
y_train = np.load('../MAT/train_y.npy')
y_test = np.load('../MAT/train_y.npy')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


def custom_net(input_shape):  # Custom Model
    X_input = Input(input_shape)
    X = Conv2D(10, (1, 1), strides=(1, 1), name='conv1', padding='same', activation='tanh')(X_input)
    X = MaxPooling2D((2, 2), strides=(1, 1), name='max_pool1', padding='same')(X)
    X = Conv2D(20, (1, 1), strides=(1, 1), name='conv2', padding='same', activation='tanh')(X)
    X = MaxPooling2D((1, 1), strides=(2, 2), name='max_pool2', padding='same')(X)
    X = Conv2D(40, (1, 1), strides=(1, 1), name='conv3', padding='same', activation='tanh')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool3', padding='same')(X)
    X = Conv2D(80, (1, 1), strides=(1, 1), name='conv4', padding='same', activation='tanh')(X)
    X = MaxPooling2D((1, 1), strides=(1, 1), name='max_pool4', padding='same')(X)
    X = Conv2D(160, (1, 1), strides=(1, 1), name='conv5', padding='same', activation='tanh')(X)
    X = MaxPooling2D((1, 1), strides=(1, 1), name='max_pool5', padding='same')(X)
    X = Conv2D(640, (1, 1), strides=(1, 1), name='conv6', padding='same', activation='tanh')(X)
    X = Flatten()(X)
    X = Dense(500, activation='sigmoid', name='fc1')(X)
    X = Dense(214, activation='softmax', name='fc2')(X)
    model = Model(inputs=X_input, outputs=X, name="custom_net")
    return model


cModel = custom_net(x_train.shape[1:])
cModel.compile('adadelta', 'categorical_crossentropy', metrics=['accuracy'])
history = History()
temp=[]
i=1
cModel.fit(x=x_train, y=y_train, epochs=15)



cModel.save('../Models/CNN.h5')
