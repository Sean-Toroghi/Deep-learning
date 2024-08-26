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

At a high level, a neural network is a collection of nodes where each node has an adjustable float value called weight and the nodes are interconnected as a graph to return outputs in a format that is dictated by the architecture of the network. A typical ANN is made up of: input, hidden, and output layers. Output layer could consist of one node (predict a continuous variable) or m nodes if we want to predict categorical variables with m classes. A typical function to generate output $a$ is computed by summing bias and sum of dot product of weights and inputs: $a(x,w) =  f(w_0 + \sum w_ix_i)$






__Loss function__
- continuous variable prediction: MSE 
- categorical variable prediction: binary cross-entropy, or categorical cross-entropy

__Feedforward__

a high-level strategy for coding feedforward propagation is as follows:

1. Perform a sum product at each neuron.
2. Compute activation.
3. Repeat the first two steps at each neuron until the output layer.
4. Compute the loss by comparing the prediction with the actual output.

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

Note- Softmax is usually used for computing probability of an input belonging to one of the m number of possible output classes in a given scenario. 

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

Building block in PyTorch is the tensor, which comes in different dimensions, from 1D (scalar) to nD (multi-dim matrix). Initializing a tensor is done by running `torch.tensor([x_1, x_2, ...])`. The data type of all elements within a tensor is the same. That means if a tensor contains data of different data types (such as a Boolean, an integer, and a float), the entire tensor is coerced to the most generic data type. Even if we have a boolean False with float numbers, the False will be changed to `0.`. Some of the build in functions to initialize a tensor are: 
- `torch.zeros((tuple_dim))`
- `torch.ones((tuple_dim))`
- `torch.randint(low=, high = , size = (tuple_dim))`
- `torch.randn(tuple_dim)` to generate random numbers follow a normal distribution in given dimension
- `torch.rand(tuple_dim)` to generate random numbers between 0 and 1 in given dimension

Converting between numpy array and tensor: `torch.tensor(np.array())`

__Some of the operations on tensors__
- elementwise multiplication of a scalar by tensor x: `x * scalar_value`
- element wise addition of a scalar to values in tensor x: `x.add(scalar_value)`
- reshaping tensor x:`x.view(new_shape)`
- remove a single dimension of value 1, located in dim position i:`x.squeeze(i)` or `torch.squeeze(x,i)`
- add a dimension of value 1 to position i: `x.unsqueeze(i)` or `torch.unsqueeze(x,i)`
- Alternative to `unsqueeze` we can employ `None`. Following two codes returns the same dimension (add dim with value 1 to the second position in dim):`x.unsqueeze(1)` and `x[:,None]`
- Matrix multiplication either by `torch.matmul(x,y)` or `x@y`
- Concatenation: `torch.cat([x,y], axis = )`. Here the axis defines the dim position at which we want to concatenate.
- Get max value in a tensor: `x.max()`
- Get max value along specific dimension and its index: `max_val, max_index = x.max(dim = 0)` (here along row values)
- Permute dimension of a tensor: `x.permute(new_dim_position)`
- __NOTE__ for swapping dimensions, always use `permute`. Using `view` for this purpose will cause unforeseen results.
- Other functions: abs, add, argsort, ceil, floor, sin, cos, tan, cumsum, cumprod, diag, eig, exp, log, log2, log10, mean, median, mode, resize, round, sigmoid, softmax, square, sqrt, svd, and transpose.
- Get all methods for a torch tensor: `dir(torch.Tensor)`


__Auto gradient parameter__

Torch tensor has the ability to compute gradients by specifying `requires_grad = True` when defining the input tensor. Then by calling `output.backward()` function, we get the gradient with respect to input x by calling `x.grad()` function.

Note that Pytorch is specifically optimized to run on GPu, which gives it an edge compared with computing the same outcome via numpy.

# Building a neural network with PyTorch

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
    def __init__(self): # ensure class inherits from nn.Module
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
print(f'weight shape for layer 1: {network_1.layer1.weight.shape}')
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

- Batch size is the number of data points considered to calculate the loss value or update weights. This hyperparameter helps to perform optimization when the size of the dataset is so large that it does not fit memory. The batch size helps ensure that we fetch multiple samples of data that are representative enough, but not necessarily 100% representative of the total data.
- Dataset class requires to return two values: length of the dataset, and fetch specific rows in the dataset. We pass input and output into this class.
- DataLoader gets the dataset, created by the Dataset class, and batch_size. It then is used to fetch the batch_size number of datapoints.

__Create a custom dataset and dataloar__
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

# create instance for custom dataset
dataset = CustomeDataset(x,y)

# Create dataloader to fetch the batch_size number of datapoints
train_loader = DataLoader(dataset = dataset, batch_size = 2, shuffle = True)

# get x and y from dataloader
for (x,y) in train_loader:
  print(f'x: {x}, y: {y}')
```


### All together: Custom dataset, dataloder, nn model, and make prediction
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

# ------------------ Create custom dataset ------------------
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

# create instance for custom dataset
dataset = CustomeDataset(x,y)

# Create dataloader to fetch the batch_size number of datapoints
train_loader = DataLoader(dataset = dataset, batch_size = 2, shuffle = True)

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


### Create custom loss function
A Custom Loss function can be created by defining a function that gets the $y$ and $\hat{y}$ and returns a value representing the computed loss.

```python
def mean_squared_error_function(y_hat, y):
    loss = (y_hat - y)**2
    loss = loss.mean()
    return loss
```
### Get the output of an intermediate layer
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
Employ sequence method simplified building a neural network. It uses the `Sequential` class, and requires to perform the same steps as `nn.Module` to build a network. 


### Saving/loading a model
 To define a model, we need three components: 
- unique name for each parameter correspond to `__init__`
- logic to connect every tensor in the network to one another correspond to `forward`
- a value (weight/bias) of each tensor correspond to the updated weight/bias during training

Employ  `model.state_dict()` is used to save/load a model. It returns a dictionary, in which keys are the names of the model's layers, values are the weights of the layers. Note to send model to cpu before initializing the save method. This way we save cpu tensors, and can later load them even if cuda is not available. 

When loading a saved mode, we need to first build a model with the exact same architecture as the saved model, and assign the saved values to it.

__Note:__ although an alternative method for saving a model is to save its architecture and parameters together via invoking `torch.save(model, '<path>')` and load it later via `torch.load(model,'<path>')`, it is not advisable. In case the torch version changes, we won't be able to run it (incompatible torch version between saved and load models).

__Example: building a toy model with Sequential method__
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import time
import matplotlib.pyplot as plt
from torchsummary import summary
# ------------------- Dataset ----------------------
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
Neural network models are very capable with regards to tasks that have images as inputs, including image classification, object detection, image segmentation, and many other emerging ones. 

In traditional computer vision, a range of methods were used for feature extraction. However, the drawback to this approach is it requires an expert in image and signal analysis. Some of the feature extraction methods are:
- histogram feature: useful for tasks such as auto-brightness, or night vision.
- edge and corners feature: for tasks such as image segmentation.
- color separation feature: in tasks such as traffic light detection, a model is required to detect different colors.
- image gradients feature: a step further of color separation feature that is aimed to understand how the colors change at the pixel level. It also acts as a prerequisite for edge detection.
 

### Representing an image
A digital image is represented by an array of pixels, each has a value between 0 to 255 for black and white image, and three dim vectors of pixels, one for each channel of RGB for a color image. 

Image dimension (height, width, c) corresponds to (row, column, channels). It can be converted into structured arrays and scalars, and shown by employing `cv2` and `matplotlib` libraries. Also different preprocessing, such as cropping can be implemented once the image is loaded as an array. 

```python
import cv2, matplotlib.pyplot as plt
image = cv2.imread('<path>')

# apply preprocessing ---------------
# crop
image = image[50:250, 40:240]
# convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# plot the image
plt.imshow(image, cmap = 'gray)
```

__Example of a RGB image of size 3x3__

<img src="https://github.com/user-attachments/assets/b56857ad-3844-49eb-ab09-75f99d88579a" width="150" height="150">
 
## Image classification with deep learning 
Steps that are required to build and train a model:
- Import the relevant packages
- Build a dataset that can fetch data one data point at a time
- Wrap the dataloader from the dataset
- Build a model and then define the loss function and the optimizer
- Define two functions to train and validate a batch of data, respectively
- Define a function that will calculate the accuracy of the data
- Perform weight updates based on each batch of data over increasing epochs


## Effect of hyperparameters and model design
- __batch size__: larger batch size negatively affects the performance. However, the smaller batch size results in a longer training time. Each epoch requires more runs. In summary, having a lower batch size generally helps in achieving optimal accuracy when we have a small number of epochs, but it should not be so low that training time is impacted.
- __optimizer function__: optimizer drives weights to optimal value at which the loss will be minimal. Some of the optimizer functions are Adam, SGD, Adagrad, Adadelta, AdamW, LBFGS, and RMSprop.
- __deeper neural network model__: as the model gets deeper, its complexity increases. This leads to overfitting.
- __input normalization__: when the input value is large, the variation of the sigmoid output doesn’t make much difference when the weight values change considerably. As a result, and to avoid its negative effect on accuracy, we need to normalize inputs, prior to feeding them into the model. similar to a large value as input, but at the other side of spectrom, when the input values are very small, the sigmoid output changes slightly, requiring a big change to the weight value to achieve optimal results.
- __batch normalization__: similar to input normalization, values in hidden layers could get very large or very small, which negatively affect the model to correctly learn and predict. Batch normalization is performed by computing batch norm and standard deviation and then normalizing the batch values by subtracting each value from the batch mean and dividing by the batch variance (hard normalization). In soft normalization, the network identifies best normalization parameters: $\alpha , \beta$.
- __dropout__: helps to reduce risk of overfitting. Dropout is a mechanism that randomly chooses a specified percentage of node activations and reduces them to 0. In the next iteration, another random set of hidden units is switched off. This way, the neural network does not optimize for edge cases, as the network does not get that many opportunities to adjust the weight to memorize for edge cases. NOTE: during prediction, dropout doesn’t need to be applied.
- __regularization__: one feature of overfitting is that some of the weights become super large during the training. To prevent this from occurring, we can employ regularization. This technique penalizes the model for having large weight values. There are two types of regularization: L1 and L2. Regularization is incorporated into a model, during the training steps, by adding the penalty term when computing the loss in forward pass.
  - L1: regularization ensures that it penalizes for the high absolute values of weights by incorporating them in the loss value calculation.
  ```python
  model.train()
  prediction = model(x)
  l1_regularization = 0
  for param in model.parameters():
      l1_regularization += torch.norm(param,1)
  batch_loss = loss_fn(prediction, y)+0.0001*l1_regularization
  batch_loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  ```
  - L2: it penalizes large weight values by having the sum of squared values of weights incorporated into the loss value calculation. Similar to L1, L2 term is added to the loss during the forward pass
  ```python
  model.train()
  prediction = model(x)
  l2_regularization = 0
  for param in model.parameters():
      l2_regularization += torch.norm(param,2)
  batch_loss = loss_fn(prediction, y) + 0.01*l2_regularization
  batch_loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  ```
__Example__: image classification


---

# Convolutional neural network and its application in image classification, object classification and detection


__Concepts and terms__
- convolution: A convolution is basically a multiplication between two matrices. However, it is not exactly like how we multiply two matrices in algebra. In nutshell, convolution is element wise multiplication, and for two different size matrices, we slide the smaller matrix over the large one, and perform the element wise multiplication. The smaller matrix is called _filter_ or _kernel_, and bigger matrix is the original image.
- filer (or kernel): is a matrix of weights that is initialized randomly at the start. The model learns the optimal weight values of a filter over increasing epochs. As the number of filters increases, the model gains more capability to learn from an image ( the filters learn about different features present in the image). If we employ $n$ filters, the output will be a matrix with $n$ channels.
 


  <img src = "https://github.com/user-attachments/assets/b32ed609-6e81-415d-a901-c05a52158407" width="300" height="250">

- stride, padding, ad pooling
- flatten layer (fully connected layer) 

The operations of convolution and pooling constitute the feature learning section, as filters help in extracting relevant features from images and pooling helps in aggregating information and thereby reducing the number of nodes at the flatten layer.


<img src = "https://github.com/user-attachments/assets/594f6872-c905-400b-bc8e-fddf297aadc1" width="350" height="250">

Convolution and pooling can also help us with the __receptive field.__ 


---
# Transfer learning for image classification task

__Transfer learning__, in general, is a technique aims to improve model performance by transfering learning of the model on a generic dataset to the specific task in hand. This technique leverages the gained knwolesge to solve another smiliar task.

__Transfer learning - high level__
1. Normalize the input images, normalized by the same mean and standard deviation that was used during the training of the pretrained model.
2. Fetch the pretrained model’s architecture. Fetch the weights for this architecture that arose as a result of being trained on a large dataset.
3. Discard the last few layers of the pretrained model. They are replaced by new layers for fine-tune task.
4. Connect the truncated pretrained model to a freshly initialized layer (or layers) where weights are randomly initialized. The output of the last layer needs to have as many neurons as the classes/outputs of the task in hand.
5. Freeze the pretrained model layes (prevent them from updating their weights) and train/update weights of the newly initialized layer and the weights connecting it to the output layer are trainable.
6. Update the trainable parameters over increasing epochs to fit a model.



 Two models that are trained on ImageNet dataset (consists of 14M images and 1k classes) are VGG and ResNet. 

 ## VGG16

Visual geometry group was developed by the University of Oxford with 16 layers. It was trained on ImageNet and was the runner-up model in 2014. To get the pretrained VGG16, we can simply upload it from `torchvision.models` library: `models.vgg16(pretrained=True)`. It has about 138M parameters, with 13 conv/pool layers and 3 linear layers. VGG has other variations (VGG11 and VGG19). However, increasing the number of layers without having skip/residual connection does not improve model performance.

<img src = "https://github.com/user-attachments/assets/1099b2d4-aee1-4138-bb96-6a9140fa1aea" width="350" height="250"> [Ref,](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781803231334/files/Images/B18457_05_01.png)


## Fine-tune
Ther are three major sub-modules in the model:
1. features
2. avgpool
3. classifier

To fine-tune the model, typically the first two sub-modules are freezed and we delete the last layer (classifier) and replace it with a new layers according to the task in hand (number of classes). 

### Transfer-learning VGG16
1. normalize the dataset: it is mandatory to resize, permute, and then normalize images. The input images require to be scaled to a value between 0 and 1, having mean of [0.485, 0.456, 0.406] and sd of [0.229, 0.224, 0.225]. Define dataloader for trian and test sets.
2. Getting model arch and weights
3. Decide which layers to keep (freeze) and which one/s to discard (`avgpool` and `classifier`)
4. Define new layers to be replaced by the discard ones.
5. Define loss function and optimizer, and initialize the model
6. Define training and accuracy functions
7. Perform fine-tuning and trace the model performance

## ResNet

For deeper models, to address the problem of vanishing gradient (during backpropogation) and vanishing information about the input in the last layers during the forward pass, ResNet architecture was introduced. It employs a skip connection in its residual block that passes information directly to the output.

<img src = "https://github.com/user-attachments/assets/7f783df5-c8b6-448e-b651-6636f3811d56" width="250" height="250"> [Ref,](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781803231334/files/Images/B18457_05_05.png)


<img src = "https://github.com/user-attachments/assets/04194da5-1b65-447b-95c0-5ec58f9322b9" width="350" height="250"> [Ref,](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781803231334/files/Images/B18457_05_06.png)


### Implementing ResNet

To define the residual block, we need to add padding to gurantee having same dimension when performing the adding function. 


ResNet18 has the following components, and to preform fine-tuning we can freeze all layers except the last two layers:

- Convolution
- Batch normalization
- ReLU
- MaxPooling
- Four layers of ResNet blocks
- Average pooling (avgpool)
- A fully connected layer (fc)












---
# Object detection


## Intro: object detection task
Object detection, in general terms, refers to the task of detecting an object for tasks such as classification, or drawing a boundary around an object in an image. Object detection task could be divided into two forms:
- object localization: draw a tight bounding box around an object
- object detection: the general form of object detection that creates bounding boxes around objects in an image and identify each object (predict its class).

Two methods for identifying regions surrounding an object are `SelectiveSearc` and anchor boxes. 

Some applications for object detection task are: 
- security: identify personnel in surveillance cameras vs intruders.
- Autonomous vehicles: recognizing the various objects present in the image of the surrounding.
- Image searching: identify the images containing an object (or a person) of interest.
- Automotives: identifying a number plate within the image of a car or an id card number.

An example i s performing image classification when there is a cat and dog in the image. The model not only needs to identify the location of the two objects, but also predict the classes associated with the two objects. 

A high level steps for object detection is as follow:
1. Creating ground-truth data that contains labels of the bounding box and class corresponding to various objects present in the image
2. Coming up with mechanisms that scan through the image to identify regions (region proposals) that are likely to contain objects
3. Creating the target class variable by using the IoU metric
4. Creating the target bounding-box offset variable to make corrections to the location of the region proposal in step 2
5. Building a model that can predict the class of object along with the target bounding-box offset corresponding to the region proposal
6. Measuring the accuracy of object detection using mean average precision (mAP)


### Creating a bounding-box ground truth for training
Creating input-output pairs, where the input is the image and the output is the bounding boxes and the object classes, is the first step in the object detection pipeline. The bounding box is in practice performed by identifying four pixels that identify the four corners of the box. One approach is to employ [ybat](https://github.com/drainingsun/ybat) to create (annotate) and store bounding boxes around objects in the image in XML format. 

### identify region proposals
Region proposal is a technique that helps identify islands of regions where the pixels are similar to one another. This technique helps to identify location of objects presented in an image. Additionally, given that a region proposal generates a proposal for a region, it aids in object localization where the task is to identify a bounding box that fits exactly around an object.

__SelectiveSearch__

SelectiveSearch is a region proposal algorithm used for object localization, where it generates proposals of regions that are likely to be grouped together based on their pixel intensities. SelectiveSearch groups pixels based on the hierarchical grouping of similar pixels, which, in turn, leverages the color, texture, size, and shape compatibility of content within an image. The output is a segmented image with proposal regions. 

```python
import selectivesearch
img = cv2.imread(img_path)
img_fz = felzenszwalb(img, scale=200) # scale represents the number of clusters that can be formed within the segments of the image

# define a function to extract proposal regions
def extract_candidates(img):
  img_lbl, regions = selectivesearch.selective_search(img, scale=200,  min_size=100)
  img_area = np.prod(img.shape[:2])
  candidates = []
  for r in regions:
      if r['rect'] in candidates: continue
      if r['size'] < (0.05*img_area): continue
      if r['size'] > (1*img_area): continue
      x, y, w, h = r['rect']
      candidates.append(list(r['rect']))
    return candidates
#example
candidates = extract_candidates(img)
```

Next step is to compute the intersection of a region proposal candidate with a ground-truth bounding box.

### Intersection over union (IoU)
Intersection refers to measuring how much the predicted and actual bounding boxes overlap, while union refers to measuring the overall space possible for overlap. IoU is the ratio of the overlapping region between the two bounding boxes over the combined region of both bounding boxes.

```python
# function for computing IoU for two nput boxes
def compute_IoU(boxA, boxB , epsilon=1e-5):
  # compute coordinates
  x1 = max(boxA[0], boxB[0])
  y1 = max(boxA[1], boxB[1])
  x2 = min(boxA[2], boxB[2])
  y2 = min(boxA[3], boxB[3])

  # compute width and height
  width = (x2 - x1)
  height = (y2 - y1)
  # compute overlap area
  if (width<0) or (height <0):
    return 0.0
  area_overlap = width * height

  # compute combined area
  area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  area_combined = area_a + area_b - area_overlap

  iou = area_overlap / (area_combined+epsilon)
  return iou
```

### Non-max suppression
To identify and pick the region among many region proposals, we can employ non-max suppression. Non-max refers to the boxes that don’t have the highest probability of containing an object, and suppression refers to us discarding those boxes. In non-max suppression, we identify the bounding box that has the highest probability of containing the object and discard all the other bounding boxes that have an IoU below a certain threshold with the box showing the highest probability of containing an object. In PyTorch, non-max suppression is performed using the `nms` function in the `torchvision.ops` module.
 

### Mean average precision
mAP is the average of precision values calculated at various IoU threshold values across all the classes of objects present within a dataset.


## Training R-CNN-based and Fast R-CNN-based custom object detectors

R-CNN stands for region-based convolutional neural network. Region-based within R-CNN refers to the region proposals used to identify objects within an image.

### R-CNN workflow
To detect an object R-CNN performs the following steps:
1. Extract region proposals from an image. We need to ensure that we extract a high number of proposals to not miss out on any potential object within the image.
2. Resize (warp) all the extracted regions to get regions of the same size.
3. Pass the resized region proposals through a network. Typically, we pass the resized region proposals through a pretrained model, such as VGG16 or ResNet50, and extract the features in a fully connected layer.
4. Create data for model training, where the input is features extracted by passing the region proposals through a pretrained model. The outputs are the class corresponding to each region proposal and the offset of the region proposal from the ground truth corresponding to the image.
5. Connect two output heads, one corresponding to the class of image and the other corresponding to the offset of region proposal with the ground-truth bounding box, to extract the fine bounding box on the object.
6. Train the model after writing a custom loss function that minimizes both the object classification error and the bounding-box offset error.

  <img src = "https://github.com/user-attachments/assets/d9ce00a3-c0fd-4308-a146-49990a5fdd6b" width="550" height="150"> [Ref.](https://arxiv.org/pdf/1311.2524.pdf)


### Implementing R-CNN for object detection


































