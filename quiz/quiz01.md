# Quiz 1 - Introduction to deep learning

## What is the role of a neural network's activation function?

The activation function introduces non-linearity into the network, allowing it to learn more complex patterns.

## Why is the sigmoid function often used as the activation function in neural networks?

The sigmoid function is often used as it has a smooth gradient and its output values are bound between 0 and 1, which is especially useful for binary classification problems.

## What does the feedforward step in a neural network do?

The feedforward step in a neural network computes the output of the network given an input by propagating the input data through the layers of the network.

## What is the purpose of the loss function in training a neural network?

The loss function measures the difference between the network's predictions and the actual outputs, and the goal of training is to minimize this loss.

## Why do we use the mean squared error as the loss function?

The mean squared error is used in this tutorial because it's simple to understand and its derivative is easy to compute, making it a good choice for regression problems.

## What is the derivative of the mean squared error loss function with respect to the predicted output?

The derivative of the mean squared error loss function with respect to the predicted output is `2/n Σ(predicted - actual)`, where n is the number of samples.

## What is the role of backpropagation in training a neural network?

Backpropagation is used to compute the gradients of the loss function with respect to the weights of the network, which are then used to update the weights.

## How does the chain rule from calculus apply to backpropagation?

The chain rule allows us to compute the derivative of the loss function with respect to the weights by multiplying the derivative of the loss with respect to the output and the derivative of the output with respect to the weights.

## How do you compute the derivative of the output of a neuron with respect to its input?

The derivative of the output of a neuron with respect to its input is the derivative of the activation function applied to the weighted sum of the inputs.

## How do you compute the derivative of the weighted sum of a neuron's inputs with respect to the weights?

The derivative of the weighted sum of a neuron's inputs with respect to the weights is just the inputs.

## How do you update the weights of a neural network during backpropagation?

The weights of a neural network are updated during backpropagation by subtracting the gradients of the loss function with respect to the weights.

## Why do we iterate over the forward propagation and backpropagation steps multiple times when training a neural network?

We iterate over the forward propagation and backpropagation steps multiple times to gradually adjust the weights of the network and minimize the loss.

## What is an epoch in the context of neural network training?

An epoch is a complete pass through the entire training dataset.

## How can you use a trained neural network to make predictions on new data?

You can use a trained neural network to make predictions on new data by performing forward propagation with the new data and the final weights.

## Why do we add an optional parameter to the feedforward method when preparing to use the network for making predictions?

We add an optional parameter to the feedforward method to allow us to use the same method for both training and making predictions with new data.

## What is the XOR problem and why is it significant in the context of neural networks?

The XOR problem is a classification problem that is not linearly separable, meaning it cannot be solved by a single-layer perceptron. It serves as a good test for the capability of a neural network to learn non-linear patterns.

## What is the impact of changing the number of neurons in the network's hidden layer?

Changing the number of neurons in the hidden layer affects the capacity of the network. More neurons allow the network to model more complex functions, but also make it more prone to overfitting.

## What is the impact of changing the number of epochs in the network's training process?

Changing the number of epochs in the training process affects how much the network learns from the data. More epochs can lead to better learning up to a point, after which the network may start overfitting to the training data.

## Why is it important to understand the principles of neural networks and backpropagation even when using a high-level library like TensorFlow or PyTorch?

Understanding the principles of neural networks and backpropagation helps you understand how these libraries work under the hood, allowing you to use them more effectively and troubleshoot issues when they arise.

## How would the network's training and prediction process change if we used a different activation function (like ReLU) or a different loss function (like cross-entropy)?

Using a different activation function or loss function would require adapting the computation of the derivatives in the backpropagation step. Different functions might be better suited to different types of problems (e.g., ReLU and cross-entropy loss are often used in classification problems).
