# Linear layer of a neural network

A linear layer in a neural network is a function that applies a linear transformation to its inputs. It's implemented as a matrix multiplication and a bias offset: 

```
y = x * W.T + b
```

x is the input,
W is the weight matrix (also referred to as the "parameters" of the layer) and it is transposed to make it suitable for matrix multiplication, and
b is the bias.

The weight and bias are learnable parameters adjusted during training. During training, these parameters are adjusted with backpropagation to minimize the output error of the neural network.
The layer is fully connected, meaning each input neuron is connected to each output neuron. 

It's often used with non-linear activation functions to capture complex patterns in data. In PyTorch, a linear layer is created with nn.Linear, specifying input size (in_features) 
and output size (out_features).

The pytorch `nn.Linear` takes two required arguments:

in_features: the size of each input sample (i.e., the number of input features)
out_features: the size of each output sample (i.e., the number of output features)
You create a linear layer in PyTorch like this:

```python
import torch.nn as nn

fc = nn.Linear(in_features=10, out_features=5)
```

Here, `fc` is a fully connected or linear layer. It takes in a vector of size 10 and outputs a vector of size 5. The actual transformation it applies to the data 
will depend on the values of the weight matrix and bias, which are randomly initialized and then learned through training.

In the context of neural networks, these linear layers are often stacked with activation functions in between them, creating a multi-layer perceptron (MLP) model.




