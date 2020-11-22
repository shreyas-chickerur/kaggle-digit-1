# Classify MNIST handwritten digit images as 0 - 9

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import mnist

# Load the data
train_images = mnist.train_images() # training data images - stored in numpy arrays
print("The initial size of the training images: ", train_images.shape) # 60k
train_labels = mnist.train_labels() # training data labels
print("The number of training labels is: ", train_labels.shape) # 60k

test_images = mnist.test_images() # testing data images
print("The initial size of the testing images: ", test_images.shape)
test_labels = mnist.test_labels() # testing data labels
print("The initial size of the testing labels: ", test_labels.shape)


# Normalize the images - normalize pixel values from [0, 255] to [-0.5, 0.5] to make our network easier to train
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images - flatten each 28 x 28 image into a 784 dimensional vector to pass to NN
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1,784))
# Print the shape
print(train_images.shape) # 60,000 rows and 784 columns
print(test_images.shape) # 10,000 rows and 784 columns

# Build the model
# 3 layers, 2 layers with 64 neurons and the relu function
# 1 layer with 10 neurons and softmax function
model = Sequential([
    Dense(64, activation='relu', input_dim=784),
    Dense(250, activation='sigmoid'),
    Dense(250, activation='sigmoid'),
    Dense(250, activation='sigmoid'),
    Dense(250, activation='sigmoid'),
    Dense(10, activation='softmax')
])
'''
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
'''
# Compile the model
# The loss function measures how well the model did on training and then tries to improve on it using the optimizer
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # classes that are greater than 2
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_images,
    to_categorical(train_labels), # ex for 2, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    epochs=5, # number of iterations over the entire dataset to train on
    batch_size=32 #number of samples per gradient update for training
)

# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)

# save the model
model.save_weights('model.h5')

# predict on the first 5 test images
predictions = model.predict(test_images[:5])
# print our models predictions
print(np.argmax(predictions, axis=1))
print(test_labels[:5])

# view the images
for i in range(0, 5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
