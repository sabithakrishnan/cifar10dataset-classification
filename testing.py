import tensorflow as tf  
import pickle
import numpy as np
import matplotlib.pyplot as plt
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()
files=open("D:/prognica/AI Assignment_Sabitha Kishnadhas/task1/cifartrainingmodel.pkl",'rb')
model = pickle.load(files)
model.summary()
image_number =4
plt.imshow(x_test[image_number])
 
# load the image in an array
n = np.array(x_test[image_number])
 
# reshape it
p = n.reshape(1, 32, 32, 3)
 
# save the predicted label
predicted_label = labels[model.predict(p).argmax()]
 
# load the original label
original_label = y_test[image_number]
original_label=labels[original_label]
 
# display the result
print("Original label is {} and predicted label is {}".format(
    original_label, predicted_label))
