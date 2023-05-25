# Some common pytorch methods

## torch.empty
This function creates an un-initialized tensor. It does not contain definite known values before it is used.

```python
# Creates a 3x3 empty tensor
x = torch.empty(3, 3)
print(x)
```

## torch.zeros
This function creates a tensor filled with zeros.

```python
# Creates a 3x3 tensor filled with zeros
x = torch.zeros(3, 3)
print(x)
```
## torch.ones
This function creates a tensor filled with ones.

```python
# Creates a 3x3 tensor filled with ones
x = torch.ones(3, 3)
print(x)
```
## torch.randn
This function creates a tensor with elements picked randomly from a normal distribution, with a mean of 0 and a standard deviation of 1.

```python
# Creates a 3x3 tensor with elements from a standard normal distribution
x = torch.randn(3, 3)
print(x)
```

## torch.tensor
This function creates a tensor from data. The data can be a list or a NumPy array.

```python
# Creates a tensor from a list
x = torch.tensor([1, 2, 3])
print(x)

# Creates a tensor from a 2D list
y = torch.tensor([[1, 2], [3, 4]])
print(y)
```

## torch.arange
This function creates a 1D tensor of size end-start, with elements from start to end with a step step.

```python
# Creates a tensor from 0 to 4
x = torch.arange(5)
print(x)

# Creates a tensor from 1 to 4
y = torch.arange(1, 5)
print(y)

# Creates a tensor from 1 to 8 with a step of 2
z = torch.arange(1, 9, 2)
print(z)
```
## torch.linspace
This function creates a 1D tensor of size steps with elements from start to end spaced evenly.

```python
# Creates a tensor with 5 steps from 0 to 1
x = torch.linspace(0, 1, 5)
print(x)
```
## torch.eye
This function creates an identity matrix (a 2D tensor with ones on the diagonal and zeros elsewhere).

```python
# Creates a 3x3 identity matrix
x = torch.eye(3)
print(x)
```

## Automatic Differentiation with Autograd

PyTorch uses a module called Autograd to calculate gradients. This is extremely useful for backpropagation in neural networks.

To use Autograd, we first need to create a tensor and set its requires_grad attribute to True.

```python
# Create a tensor and set requires_grad=True to track computation with it
x = torch.tensor([3.], requires_grad=True)
```

Now, let's define a simple operation.

```python
y = x * 3 + 2
```

We can call .backward() on the y tensor to calculate the gradients.

```python
y.backward()
```

And to view the gradients, we can inspect the .grad attribute of the x tensor.

```python
print(x.grad)  # This will output tensor([3.]), as dy/dx = 3
```
