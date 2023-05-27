# Tensor Basics

In this section we are going to get introduced to the basics of PyTorch. We will learn how to create tensors, perform basic operations on them, and convert them to NumPy arrays and back.

## Step 1: Installation

First, we need to install PyTorch. You can do this using pip:

```bash
pip install torch
```

## Step 2: Importing PyTorch

Next, we will import the PyTorch module in our Python script and make sure it is working:

```python
import torch
```

## Step 3: Creating Tensors

Tensors are a generalized version of matrices and are the fundamental building blocks of PyTorch.

To create a tensor in PyTorch, you can use the torch.Tensor() function:

```python
import torch

# Creating a 1D Tensor
a = torch.Tensor([1,2,3])
print(a)

# Creating a 2D Tensor
b = torch.Tensor([[1,2,3],[4,5,6]])
print(b)
```

## Step 4: Tensor Attributes

Tensors have attributes like shape, dtype, and device which they are stored in.

```python
# Shape of a tensor
print(b.shape) # or b.size()

# Data type of a tensor
print(b.dtype)

# Device the tensor is stored on
print(b.device)
```

## Step 5: Tensor Operations

We can perform various operations on tensors.

```python
# Addition
c = torch.Tensor([1,2,3])
d = torch.Tensor([4,5,6])
print(c+d)

# Multiplication (Element-wise)
print(c*d)

# Matrix Multiplication
e = torch.Tensor([[1,2],[3,4]])
f = torch.Tensor([[5,6],[7,8]])
print(e.matmul(f)) # or torch.mm(e, f)
```

## Step 6: Changing Tensor Shape

You can change the shape of a tensor without changing its data using the view() method.

```python
g = torch.Tensor([1,2,3,4,5,6])
print(g.view(2,3)) # Reshapes g to a 2x3 matrix
```

## Step 7: Converting between NumPy Arrays and PyTorch Tensors

We can easily convert a NumPy array to a PyTorch tensor and vice versa.

```python
import numpy as np

# Creating a NumPy array
numpy_array = np.array([1, 2, 3, 4, 5])

# Converting the NumPy array to a PyTorch tensor
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

# Converting a PyTorch tensor to a NumPy array
numpy_from_tensor = tensor_from_numpy.numpy()
print(numpy_from_tensor)
```

These basics will set the foundation for your learning of more advanced PyTorch concepts and operations.
