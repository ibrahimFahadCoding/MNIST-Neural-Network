#Import modules

from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

#Get images and labels for model

images, labels = get_mnist()
images_training = images[0:50000]
labels_training = labels[0:50000]
images_testing = images[50000:60000]
labels_testing = labels[50000:60000]

#Initialize Weights and Biases

w_i_h = np.random.uniform(-0.5, 0.5, (64, 784))
#Here, the input layer has 784 neurons and the hidden layer has 20, resulting in 784x20 connections between -0.5 and 0.5
w_h_o = np.random.uniform(-0.5, 0.5, (10, 64))
b_i_h = np.zeros((64, 1))
b_h_o = np.zeros((10, 1))

# [!] Important Note: size of weight matrix = (layer on right, layer on left) [!]

#Initialize other parameters

learn_rate = 0.1
#How much we learn
nr_correct = 0

epochs = 50
#How many times to run each image

#Actual networking
for epoch in range(epochs):
    for img, l in zip(images_training, labels_training):
        #Reshape vectors to matrices
        img.shape += (1,)
        l.shape += (1,)
        #Both img and l are vectors, not matrices, which will be a problem since we will need matrix multiplication.

        #Forward prop (input -> hidden)

        h_pre = b_i_h + w_i_h.dot(img)
        #here, h_pre is the hidden values before we apply an activation function.
        #We do matrix multiplication between the weights and the input values (img grayscale values) and add the biases.
        h = 1 / (1 + np.exp(-h_pre))
        #Here we use sigmoid activation function to ensure all hidden values are between 0 and 1.

        #Forward prop (hidden -> output)

        o_pre = b_h_o + w_h_o.dot(h)
        o = 1 / (1 + np.exp(-o_pre))

        nr_correct += int(np.argmax(o) == np.argmax(l))

        #Backprop (Output -> Hidden) [MSE Deriv]
        delta_o = o - l
        w_h_o += -learn_rate * delta_o.dot(h.T)
        b_h_o += -learn_rate * delta_o
        #Backprop (Hidden -> Input) [Sigmoid Deriv]
        delta_h = w_h_o.T.dot(delta_o) * (h * (1 - h))
        w_i_h += -learn_rate * delta_h.dot(img.T)
        b_i_h += -learn_rate * delta_h

    print(f"Accuracy: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

#Make Predictions
while True:
    index = int(input("Enter a number (0 - 9999): "))
    img = images_testing[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h.dot(img.reshape(784, 1))
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o.dot(h)
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"This digit is probably a {o.argmax()}.")
    print(f"Actual: {np.argmax(labels_testing[index])}")
    plt.show()
