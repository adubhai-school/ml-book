# Introduction to a Simple Neural Network

In this section, we will be creating a simple neural network and train it.

First, let's import the torch.nn module, which contains neural network layers, loss functions, and optimizers.

```python
import torch.nn as nn
```

Now, we define our simple neural network. In pytorch a neural network is a subclass of `nn.Module` and usually has an `__init__` method and a `forward` method. By convention `__init__` method 
will accept the network configuration and create some or all of the objects of building blocks or inference steps those create the whole network. The `forward` method accepts input as
parameters and run inference steps to produce model output. Here is a simple one layer neural network:

```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        
        self.layer1 = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        return x
```

In the above code, nn.Linear is a linear (also known as fully connected) layer, which performs a linear transformation on the input data. nn.ReLU is the 
activation function ReLU (Rectified Linear Unit).

## Creating the Network and Processing Inputs
Now, we create an instance of our network, and process some input data.

```python
# Create the network (input size 10, hidden size 20, output size 1)
net = SimpleNet(10, 20, 1)

# Create a dummy tensor to represent input data
input_data = torch.randn(10)

# Forward pass through the network
output_data = net(input_data)

print(output_data)
```

This will produce completely randon output (may be all zeros sometimes for any input) as we have not trained our network. In the next section we will be training 
this network so that it produces output according to training.

## Training the Neural Network

First, we need to define a loss function and an optimizer.

```python
# Define a loss function - we'll use Mean Squared Error (MSE)
criterion = nn.MSELoss()

# Define an optimizer - we'll use Stochastic Gradient Descent (SGD)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # Learning rate 0.01
```
criterion is a function that will take the output of our network and the target output and compute a loss value that we want to minimize 
through optimization. optimizer is an implementation of a specific optimization algorithm - in this case, Stochastic Gradient Descent.

Let's create some dummy target data, and compute the loss for our network's output.

```python
# Create dummy target data
target_data = torch.randn(1)  # for example purposes, this can be any data

# Compute the loss
loss = criterion(output_data, target_data)

print(loss)
```
Now we'll do a backwards pass through the network, using the .backward() method to compute the gradients of the loss with respect to the network's parameters.

```python
# Zero the gradients
optimizer.zero_grad()

# Backward pass: compute gradient of the loss with respect to all the learnable parameters of the model
loss.backward()

# Update weights
optimizer.step()
```
optimizer.zero_grad() zeros the gradients, as PyTorch accumulates gradients and we need to clear them at each step. loss.backward() computes 
the gradient of the loss with respect to all of our learnable parameters, and optimizer.step() performs a single optimization step and updates the network parameters.

We now have computed the gradients of a loss function with respect to your model's parameters, and used gradient descent to update the weights 
and reduce the loss. This process - forward pass, loss computation, backward pass, and parameter update - is essentially what training a neural network consists of.
