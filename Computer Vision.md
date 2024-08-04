<h1>Computer vision - deep learning</h1>

# Deep learning for computer vision

## Overview
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

__Building a nueral network with PyTorch__



- 
- 
