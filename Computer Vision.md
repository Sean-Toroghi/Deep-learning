<h1>Computer vision with deep learning and modern methods</h1>

**References**
- []()
- []()
- []()
- []()
- []()

--- 

# Overview
An ANN is a collection of tensors (weights) and mathematical operations arranged in a way that loosely replicates the functioning of a human brain. It can be viewed as a mathematical function that takes in one or more tensors as inputs and predicts one or more tensors as outputs. The arrangement of operations that connects these inputs to outputs is referred to as the architecture of the neural network. 

At a high level, a neural network is a collection of nodes where each node has an adjustable float value called weight and the nodes are interconnected as a graph to return outputs in a format that is dictated by the architecture of the network. A typical ANN is made up of: input, hidden, and output layers. Ouput layer could consists of one node (predict a continuous variable) or m nodes if we want to predict categorcial variables with m classes. A typical function to generate output $a$ is computed by summing bias and sum of dot product of weights and inputs: $a(x,w) =  f(w_0 + \sum w_ix_i)$






__Loss function__
- continuous variable prediction: MSE 
- categorical variabel prediction: binary cross-entropy, or categorical cross-entropy

__Feedforward__

a high-level strategy for coding feedfrward propagation is as follows:

1. Perform a sum product at each neuron.
2. Compute activation.
3. Repeat the first two steps at each neuron until the output layer.
4. Compute the loss by comparing the prediction with the actual output.
5. 
```python
def feed_forward(inputs, outputs, weights):
  pre_hidden = np.dot(inputs,weights[0])+ weights[1]
  # sigmoid activation
  hidden = 1/(1+np.exp(-pre_hidden))
  pred_out = np.dot(hidden, weights[2]) + weights[3]
  mean_squared_error = np.mean(np.square(pred_out - outputs))
  return mean_squared_error
```




__Activation functions__
- $tanh (x) =  \frac{e^x - e^{-x}}{e_x + e^{-x}}$ 
- $sigmoid (x)= \frac{1}{1 + e^{-x}}$
- Sigmoid returns value $\in \[0,1 \]$, while tanh returns value $\in \[-1,1 \]$

$$ReLU = \begin{cases} 
  x & x > 0 \\ 
  0 & x \leq 0 
\end{cases}$$


```python
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def relu(x):      
    return np.where(x>0,x,0)
def linear(x):      
    return x
def softmax(x):   
    return np.exp(x)/np.sum(np.exp(x))
```

Note- Softmax is usually used for computing probability of an input belong to one of the m number of possible output classes in a given senario. 

__Loss functions__
```python
def mse(p, y):  
    return np.mean(np.square(p - y))
def mae(p, y):      
    return np.mean(np.abs(p-y))
def binary_cross_entropy(p, y):     
    return -np.mean((y*np.log(p)+(1-y)*np.log(1-p)))
def categorical_cross_entropy(p, y):        
    return -np.mean(np.log(p[np.arange(len(y)),y]))
```
__Backpropagation__

It starts with the loss value obtained in feedforward propagation and then updates the weights of the network in such a way that the loss value is minimized as much as possible.


__Gradient descnet__

## Pytorch

Building block in PyTorch is tensor, which comes in different dimensions, from 1D (scalar) to nD (multi-dim matrix). Initializing a tensor is done by running `torch.tensor([x_1, x_2, ...])`. The data type of all elements within a tensor is the same. That means if a tensor contains data of different data types (such as a Boolean, an integer, and a float), the entire tensor is coerced to the most generic data type. Even if we have a boolean False with float numbers, the False will be changed to `0.`. Some of the build in functions to initialize a tensor are: 
- `torch.zeros((tuple_dim))`
- `torch.ones((tuple_dim))`
- `torch.randint(low=, high = , size = (tuple_dim))`
- `torch.randn(tuple_dim)` to generate random numbers follow a normal distribution in given dimension
- `torch.rand(tuple_dim)` to generate random numbers between 0 and 1 in given dimension

Converting between numpy array and tensor: `torch.tensor(np.array())`

__Some of the operations on tensors__
- elementwise multiplication of a scalar by tensor x: `x * scakar_value`
- elementwise addition of a scalar to values in tensor x: `x.add(scalar_value)`
- reshaping tensor x:`x.view(new_shape)`
- remove a single dimension of value 1, located in dim position i:`x.squeeze(i)` or `torch.squeeze(x,i)`
- add a dimension of value 1 to position i: `x.unsqueeze(i)` or `torch.unsqueeze(x,i)`
- Alternative to `unsqueeze` we can employ `None`. Following two codes returns the same dimension (add dim with value 1 to the second position in dim):`x.unsqueeze(1)` and `x[:,None]`
- Matrix multiplication either by `torch.matmul(x,y)` or `x@y`
- Concatenation: `torch.cat([x,y], axis = )`. Here axis define the dim position at which we want to perform concatenate.
- Get max value in a tensor: `x.max()`
- Get max value along specific dimension and its index: `max_val, max_index = x.max(dim = 0)` (here along row values)
- Permute dimension of a tensor: `x.permute(new_dim_position)`
- __NOTE__ for swapping dimensions, always use `permute`. Using `view` for this purpose will cause unforeseen results.
- Other functions: abs, add, argsort, ceil, floor, sin, cos, tan, cumsum, cumprod, diag, eig, exp, log, log2, log10, mean, median, mode, resize, round, sigmoid, softmax, square, sqrt, svd, and transpose.
- Get all methods for a toech tensor: `dir(torch.Tensor)`


__Auto gradient parameter__

Torch tensoro has the ability to compute gradient by specifying `requires_grad = True` when defining the input tensor. Then by calling `output.backward()` function, we get the gradient with respect to input x by calling `x.grad()` function.

Note that Pytorch is specifically optimized to run on GPu, which gives it an edge compare with computing the same outcome via numpy.

# Building a nueral network with PyTorch

## Building a simple neural network
Building a neural network requires to define the following components:
- The number of hidden layers
- The number of units in a hidden layer
- Activation functions performed at the various layers
- The loss function that we try to optimize for
- The learning rate associated with the neural network
- The batch size of data leveraged to build the neural network
- The number of epochs of forward- and backpropagation

Building a simple neural network with one hidden layer
```python

# ------------------ define tensors: input and output ------------------
import torch
x = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
y = torch.tensor([[3.],[7.],[11.],[15.]])

# send to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
y = y.to(device)

# ------------------ define neural network ------------------
import torch.nn as nn
class nn_model(nn.Module): # 
    def __init__(self): # ensure class inherets from nn.Module
        super().__init__()
        self.layer1 = nn.Linear(2,8)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(8,1)
    def forward(self,x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
        
# ------------------ Create model ------------------
network_1 = nn_model()
network_1.to(device)


# get weight and bias of each layer
print(f"weights for layer 1: {network_1.layer1.weight} ; bias for layer 1: {network_1.layer1.bias}")
print(f'weight shaoe for layer 1: {network_1.layer1.weight.shape}')
# Get all parameters of a network
print('\n\nAll parameters in the network\n\n')
for par in network_1.parameters():
  print(par)


# ------------------ compute loss ------------------
loss_fn = nn.MSELoss()
y_hat = network_1(x)
loss = loss_fn(y_hat,y)
print('initial loss:', loss)

# ------------------ optimize via sgd ------------------
from torch.optim import SGD
optimizer = SGD(network_1.parameters(), lr = 0.001)
 
# ------------------ train network ------------------
Losses = []
for _ in range(30):

  # flush previous epoch's gradient
  optimizer.zero_grad()
  # Compute loss
  loss_value = loss_fn(network_1(x),y)
  # Backpropogation
  loss_value.backward()
  # updates the weights
  optimizer.step()
  Losses.append(loss_value.item())

# ------------------ plot losses ------------------
import matplotlib.pyplot as plt
plt.plot(Losses)
plt.show();
```

### Dataset, DataLoadre, and batch size

- Batch size is the number of data points considered to calculate the loss value or updateweights. This hyperparameter helps to perform optimization when the size of dataset is so large that it does not fit memory. The batch size helps ensure that we fetch multiple samples of data that are representative enough, but not necessarily 100% representative of the total data.
- Dataset class requires to return two values: lenght of the dataset, and fetch specific row in dataset. We pass input and output into this class.
- DataLoader gets the dataset, created by Dataset class, and batch_size. It then is used to fetch the batch_size number of datapoints.

__Create a custome dataset and dataloar__
```python
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

# Toy dataset
x = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
y = torch.tensor([[3.],[7.],[11.],[15.]])

# send to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
y = y.to(device)

# Create customdataset
class CustomeDataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x, dtype = torch.float32)
    self.y = torch.tensor(y, dtype = torch.float32)

  def __len__(self):
    # specify len of dataset
    return len(self.x)
  
  def __getitem__(self,index):
    # fetch a specific row
    return self.x[index], self.y[index]

# creat instance for custom dataset
dataste = CustomeDataset(x,y)

# Create dataloader to fetcccch the batch_size number of datapoints
train_loader = DataLoader(dataset = dataste, batch_size = 2, shuffle = True)

# get x and y from dataloader
for (x,y) in train_loader:
  print(f'x: {x}, y: {y}')
```


### All together: Custome dataset, dataloder, nn model, and make prediction
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import time
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------
# ------------------ define input and output - dataset ------------------

x = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
y = torch.tensor([[3.],[7.],[11.],[15.]])

# send to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
y = y.to(device)

# ------------------ Create customdataset ------------------
class CustomeDataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x, dtype = torch.float32)
    self.y = torch.tensor(y, dtype = torch.float32)

  def __len__(self):
    # specify len of dataset
    return len(self.x)
  
  def __getitem__(self,index):
    # fetch a specific row
    return self.x[index], self.y[index]

# creat instance for custom dataset
dataste = CustomeDataset(x,y)

# Create dataloader to fetcccch the batch_size number of datapoints
train_loader = DataLoader(dataset = dataste, batch_size = 2, shuffle = True)

# ------------------------------------------------------------------------
# ------------------ define neural network ------------------
class nn_model(nn.Module): # 
    def __init__(self):  
        super().__init__()
        self.layer1 = nn.Linear(2,8)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(8,1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
        
# ------------------ Create model ------------------
network_2 = nn_model()
network_2.to(device)


# ------------------ define loss and optimization functions ------------------
loss_fn = nn.MSELoss()
optimizer = SGD(network_2.parameters(), lr = 0.001)
 
# ------------------ train network ------------------
Losses = []
epochs = 25

start = time.time()
for _ in range(epochs):
  for data in train_loader:
    x,y = data
    optimizer.zero_grad()
    loss_value = loss_fn(network_2(x),y)
    loss_value.backward()
    optimizer.step()
    Losses.append(loss_value.item())
end = time.time()
print(f'time for training {epochs} : {end-start}')

#  plot losses ------------------
plt.plot(Losses)
plt.show();


# ------------------ make a prediction ------------------
new_x = torch.tensor([[9.,10.]])
new_x = new_x.to(device)
print (f" Prediction for input {new_x}: {network_2(new_x)}")
```


### Create custome loss function
Custome Loss function can be created by defining a function that gets the $y$ and $\hat{y}$ and returns a value representing the computed loss.

```python
def mean_squared_error_function(y_hat, y):
    loss = (y_hat - y)**2
    loss = loss.mean()
    return loss
```
### Get the output of an intermeidate layer
In the case of transfer and transfer learning, we need to obtain output values for an intermediate layer in a network. Pytorch provides two method to implement this:
- method 1: call the layer as if it is a function
- method 2: specify the layer in the `forward` method

```python
# seed
torch.manual_seed(42)

# define neural network
class nn_model(nn.Module): # 
    def __init__(self):  
        super().__init__()
        self.layer1 = nn.Linear(2,8)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(8,1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.activation(x)
        out = self.layer2(x)
        return out
# create input x
x = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
x = x.to(device)

# create model
network_3 = nn_model()
network_3.to(device)

# access to the input of layer 2 (x -> layer1 -> activation -> ? ->layer2)

# method 1: call layer as a function
layer1_output = network_3.layer1(x)
activation_output = network_3.activation(layer1_output)
print(f"input of layer 2 - (method 1): {activation_output}")

# method 2: specify in the forward method
class nn_model_modified(nn.Module): #return input to layer 2
  def __init__(self):  
    super().__init__()
    self.layer1 = nn.Linear(2,8)
    self.activation = nn.ReLU()
    self.layer2 = nn.Linear(8,1)

  def forward(self,x):
    x = self.layer1(x)
    x = self.activation(x)
    layer2_input = x
    out = self.layer2(layer2_input)
    return out, layer2_input
# create the modified network, and get input to layer 2
network_4 = nn_model_modified()
network_4.to(device)
_, layer2_input = network_4(x) # equivalent to network_4(x)[1]
print(f"input of layer 2 (method 2): {layer2_input}")
```

### Sequential method for building a neural network
Employ sequene method simplofies building a neural network. It uses `Sequencial` class, and requires to perform the same steps as `nn.Module` to build a network. 


### Saving/loading a model
 To define a model, we need three components: 
- unique name for each parameter corespond to `__init__`
- logic to connect every tensor in the network to one another correspond to `forward`
- a value (weight/bias) of each tensor correspond to the updated weight/bias during training

Employ  `model.state_dict()` is used to save/load a model. It reqturns a dictionary, in which keys are the names of the model's layers, valuyes are the weights of the layers. Note to send model to cpu before initialize the save method. This way we save cpu tensors, and can later load them even if cuda is not available. 

When loading a saved mode, we need to first build a model with the exact same architecture as the saved model, and assign the saved values to it.

__Note:__ although an alternative method for saving a model is to save its architecture and parameters together via invocking `torch.save(model, '<path>')` and load it later via `torch.load(mdoel,'<path>')`, it is not advisable. In case the torch version changes, we won't be able to run it (incompatible torch version between saved and load models).

__Example: building a toy model with Sequential method__
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import time
import matplotlib.pyplot as plt
from torchsummary import summary
# ------------------- Dataet ----------------------
X = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
y = torch.tensor([[3.],[7.],[11.],[15.]])

# send to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = X.to(device)
y = y.to(device)

# Create custom dataset class 
class CustomeDataset(Dataset):
  def __init__(self,X,y):
    self.X = torch.tensor(X, dtype = torch.float32).to(device)
    self.y = torch.tensor(y, dtype = torch.float32).to(device)
  def __getitem__(self,index):
    return self.X[index], self.y[index]
  def __len__(self):
    return len(self.X)
# define dataset and dataloader
dataset = CustomeDataset(X,y)
train_loader = DataLoader(dataset, batch_size = 2, shuffle = True)

# ------------------- Model ----------------------
network_5 = nn.Sequential(nn.Linear(2,8),
                          nn.ReLU(),
                          nn.Linear(8,1)
                          ).to(device)

# Get model summary: summary(model_name, input_size)
# since the input dim for first layer is (2,8)
# input size needs to have 2 in its second position (for matrix multiplication)
summary(network_5, (1,2))

# loss and optimization
loss_fn = nn.MSELoss()
optimizer = SGD(network_5.parameters(), lr = 0.001)

# training
losses = []
epochs = 25
for _ in range(epochs):
  for data in train_loader:
    x,y = data
    optimizer.zero_grad()
    loss_value = loss_fn(network_5(x),y)
    loss_value.backward()
    optimizer.step()
    losses.append(loss_value.item())

# plot losses
plt.plot(losses)
plt.show();

# make a prediction
new_x = torch.tensor([[8,9],[10,11],[1.5,2.5]])
new_x = new_x.to(device)
print (f" Prediction for input {new_x}: \n{network_5(new_x)}")
# ------------------------------------------------------------
# saving model
torch.save(network_5.state_dict(), 'model.pth')
# loading model
network_6 = nn.Sequential(nn.Linear(2,8),
                          nn.ReLU(),
                          nn.Linear(8,1)
                          ).to(device)
network_6.load_state_dict(torch.load('model.pth'))
```

---

## Building a deep neural network with PyTorch

### Representing an image
A digital image is reprsented by an array of pixels, each has a value between 0 to 255 for black and white image, and three dim vector of pixels, one for each channel of RBG for a color image. 

Image dimension (height, width, c) corresponds to (row, column, channels). It can be converted into a structured arrays and scalars, and shown by employing `cv2` and `matplotlib` libraries. Also different preprocessing, such as cropping can be implemented once the image is loaded as an array. 

```python
import cv2, matplotlib.pyplot as plt
iamge = cv2.imread('<path>')

# apply preprocessing ---------------
# crop
iamge = iamge[50:250, 40:240]
# convert to grayscale
iamge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# plot the image
plt.imshow(image, cmap = 'gray)
```

__Example of a RBG image of size 3x3__

<img src="https://github.com/user-attachments/assets/b56857ad-3844-49eb-ab09-75f99d88579a" width="150" height="150">
 




