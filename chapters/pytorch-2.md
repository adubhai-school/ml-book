## TODO

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
