# Simplifying Networks with nn.Sequential
`nn.Sequential` is a handy class in PyTorch that allows you to package a sequence of transformations or layers in a simpler and more readable way. 
It's a container for Modules that can be stacked together and run at the same time.

In this section, we'll construct a simple feedforward neural network using both the regular method and nn.Sequential for comparison.

## Import Necessary Libraries
```python
import torch
import torch.nn as nn
```

## Define the Network as a module
Let's first define a simple feedforward neural network without nn.Sequential.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

The network takes an input of size 10, applies a linear transformation followed by a ReLU activation, does this once more, then applies a final linear transformation.

## Define the same network with `nn.Sequential`
Now let's see how we can define the same network using nn.Sequential.

```python
net_seq = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)
```

As you can see, nn.Sequential allows us to define the same model in a more compact way.

## Using both models
Whether we defined our model using nn.Sequential or not, we can use it in the same way. Here's an example of creating a random tensor and passing it through the model.

```python
x = torch.randn(1, 10)  # Random input tensor

output = net(x)  # Output of the regular model
print(output)

output_seq = net_seq(x)  # Output of the model defined with nn.Sequential
print(output_seq)
```

And that's it! We now know how to use nn.Sequential in PyTorch to simplify our models and make our code more readable. 
This can be especially useful when your model consists of many layers that are applied in sequence.
