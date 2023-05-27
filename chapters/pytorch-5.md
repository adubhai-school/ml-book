# Using a Pretrained Network (Transfer Learning)

Transfer learning is a powerful technique that uses pretrained networks to boost the learning process. Pretrained models are networks that have been trained on a 
large benchmark dataset, and saved. These networks can be loaded and used directly, or used in part through transfer learning.

In this section, we'll use a pretrained network from PyTorch's model zoo on a simple classification task.

## Import Necessary Libraries

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
```

## Load and Transform the Dataset

We'll use CIFAR10 dataset, and apply some transformations to it.

```python
transform = transforms.Compose(
    [transforms.Resize(224),  # Resize images to 224x224, the input size expected by many pretrained models
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize with mean and standard deviation for each channel

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
```

## Load the Pretrained Network
We'll use the ResNet-18 model, a relatively small ResNet architecture that still performs quite well.

```python
resnet18 = torchvision.models.resnet18(pretrained=True)
```

## Modify the Last Layer
ResNet-18 was trained on ImageNet, which has 1000 classes. However, CIFAR10 only has 10 classes. We need to replace the last layer with a new, untrained layer with only 10 outputs.

```python
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)
```

## Define Loss Function and Optimizer
We'll use Cross Entropy Loss and Stochastic Gradient Descent with momentum.

```python

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
```

## Train the Network
Finally, we can train the network. We'll only do a single epoch for brevity.

```python

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'Epoch {epoch+1}, mini-batch {i+1}, loss: {running_loss / 2000}')
            running_loss = 0.0

print('Finished Training')
```

And that's it! We've performed transfer learning with a pretrained network in PyTorch. From here, we could add more epochs, adjust the learning rate, or change other hyperparameters. 
