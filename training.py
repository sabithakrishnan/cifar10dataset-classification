import tensorflow as tf  
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
cifar10 = tf.keras.datasets.cifar10
# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train, x_test = x_train / 255.0, x_test / 255.0
 
# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()
fig, ax = plt.subplots(5, 5)
k = 0
 
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k], aspect='auto')
        k += 1
K = len(set(y_train))
 
#Building AI model(Preproceesing, feature extraction, feature reduction all happens here)
# input layer
inp = Input(shape=x_train[0].shape)
layer = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
layer = BatchNormalization()(layer)
layer = MaxPooling2D((2, 2))(layer)
 
layer = Conv2D(64, (3, 3), activation='relu', padding='same')(layer)
layer = BatchNormalization()(layer)
layer = MaxPooling2D((2, 2))(layer)
 
layer = Conv2D(128, (3, 3), activation='relu', padding='same')(layer)
layer = BatchNormalization()(layer)
layer = MaxPooling2D((2, 2))(layer)
 
layer = Flatten()(layer)
layer = Dropout(0.2)(layer)
 
# Hidden layer
layer = Dense(1024, activation='relu')(layer)
layer = Dropout(0.2)(layer)
 
# last hidden layer i.e.. output layer
layer = Dense(K, activation='softmax')(layer)
model = Model(inp, layer)
 
# model description
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
x_train_crop=x_train[0:10]
y_train_crop=y_train[0:10]
model.fit( x_train, y_train, epochs=10)
import pickle
model_pkl_file = "cifartrainingmodel.pkl"  
with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)