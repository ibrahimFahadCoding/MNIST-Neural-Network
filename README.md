# MNIST-Neural-Network
This is a project where I develop a neural network that detects the digit depicted in the 28x28 image.

# How will it work?

## Step 1: Initialize Weights and Biases

To do this, we create matrices with the weights that will connect the layers. We do this by creating a matrix with the dimensions (# neurons in layer, # neurons in previous layer) and filling it up with random numbers (usually between -0.5 and 0.5)
Example: If the input layer has 3 neurons and the hidden layer has 4 neurons, the matrix sie would be (4, 3) and if the output has 2 neurons the matrix for the weights between the hidden and output layers would be (2, 4)

Then we create a matrix of biases for the hidden layers and the output layers and set every value in it to 0.
Example: If the hidden layers has 4 neurons, our biases matrix would be of size (4, 1).

## Step 2: Forward Propogation

First, we begin our training loop by changing the shape of the image and labels from vectors to matrices.

After this, we do matrix multiplication between the input layer values and the hidden layer, then add the biases. 
This is still the pre-activated values for the hidden layer, as we then need to apply an activation function, which in this case, is sigmoid.
We do the same thing with the hidden and output layer values.

## Step 3: Backward Propogation

Now, we have gotten our output, but we have to analyze the output and see how much error there is and how each neuron contributed to the error. Here, we use the derivative of the activation function and multiply it by the error of the previous layer.

## Step 4: Predict

We train our model 50 times for all 55000 training images, and once we are done with training, we can then proceed directly yo predicting test images, of which there are 5000.



