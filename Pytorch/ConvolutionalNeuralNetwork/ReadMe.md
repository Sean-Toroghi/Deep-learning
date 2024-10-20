# Deep learning with Pytorch: Convolution neural netwrork

__References__
- [Book: Mastering PyTorch - Second Edition](https://learning.oreilly.com/library/view/mastering-pytorch/9781801074308)
- [Online course: Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Online course: Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/#course-materialsoutline)

__Codes__
- [LeNet]()
- [AlexNet]()
- [VGG]()
- [GoogLeNet]()
- [ResNet and DenseNet]()
- [EfficientNets ]()


## The advantages of CNN
Convolutional neural networks are among the most powerful architecutres in deep learning field that are beign used in a range of aras from image-related tasks (such as image classification, object detection, object segmentation, and video processing) to natural language processing and speech recognition.

While CNN has several features that gives it an edge, such as weight sharing (more parameter efficient), feature extraction (learning features representation through traiing phase), hierarchcal elarning (as going deeper, the model learns low-, mid-, and high- level features), and ability to explore temporal and spatial crrelations in the data, over the past year several advancements improves it overal performance incuding:
- employ Adam for better parameter optimization,
- employ better activation and loss function, such as employ ReLU,
- intriduce batch normalization and dropout as means for regularization.
Furthermore, adopting several structural improvement to model architecture helps the model gain more power to generalize better results, including:
- employ different kernel size throughout the model helps to explore different levels of visual features in input data.
- using width-base CNN, in which width is the number of channels or feature maps in the data or features extracted from the data.
- using depth-base CNN, in which depth is number of layers. The model uses several convolutional block, each has several layers.
- employ multi-path-based CNN, in which through shortcut connetions between blocks of convolution results in better flow of information across several layers.

# CNN model architectures through history

CNN model architectures ahve evolve over the past thrity years. The initila architecture was introduced in 1989 (ConvNet) and later LeNet in 1998. With a large gap (due to lack of proper dataset, and low computational power) the next big move was the introduction of lexNet in 2012. After that over 8 years several advancements were made and each results in an introduction of a new model architecture;
- 2014: GoogLeNet, Inception, and VGG
- 2015: ResNet
- 2016: DenseNet
- 2017: ResNeXt
- 2018: Channel-boosted CNN
- 2019: EfficientNet

## LeNet
LeNet is one of the earliest CNN models, consists of 5 layers and 60k parameters. It has two block of CNN and avg pooling after each block. The there are two fully connected blocks, and one output layer that generate an output array is size 10 (number of target classes).

 __Code: implementation and train LeNet on NIST dataset (10 labels)__ []()

 ## AlexNet
 AlexNet is the successor of LeNet, which has 8 layers (5 conv layers and 3 FC layers) and 60M parameters. First two Cnv layers are each followed by Max pooling (instead of Avg pooling in LeNet). Then 3 conv layers are stacked and the output goes through another max-pooling layer, and going then through 3 FC layers it generates the final output of an array of size 1k. AlexNet is pre-trained on ImageNet with 1k labels. To use AlexNet, we can load the pre-trained model from torchvision, and fine tune it per the task in hand.

__Code: fine-tune AlexNet to classify a 2 class dataset__ []()



## VGG 
VGG13 is the next step in advancement of CNN models, with 13 layers and 138M parameters. Other variants of VGG13 are VGG16 and VGG19. There are another version of each VGG variant with added batch normalization (VGG13_bn, VGG16_bn, and VGG19_bn). 

__Code: fine-tune VGG14 to classify a 2 class dataset__ []()

## Inception / GoogLeNet

Inception/GoogLeNet uses a different architecture structure to advance the CNN architecture: multiple parallel convolutional layers. GoggLeNet (Inception v1) with 22 layers has only 5M parameters (compare with VGG with 13 layers has 138M parameters). This reduction in size is due to the following features and modifications:
- The inception module – a module of several parallel convolutional layers
- Reduce # of parameters by using 1x1 convolutions. The 1x1 conv layer does not change change the width and height of the image representation but can alter the depth of an image representation. 
- Reduces overfitting by using global average pooling instead of a fully connected layer
- Using auxiliary classifiers for training – for regularization and gradient stability. The auxilary layers are switch off during inference/prediction.

Inception v3 has 24M parameters (compare with 5M in v1) and adds a new structural feature to the original Inception model (v1): stacked sequentially. It is basically an extention of v1 architecutre. 

__Code: fine-tune Inception v1 and v3 for binary classification__ []()

## ResNet and DenseNet
Both ResNet and DenseNet come in variants such as ResNet-50, ResNet-152, DenseNet121, DenseNet161, DenseNet169, and DenseNet201,
__ResNet__

ResNet introduced the concept of skip connections, which overcomes the problem of both parameter overflow and vanishing gradients. There are two kinds of residual blocks – convolutional and identity, both having skip connections. For the convolutional block, there is an added 1x1 convolutional layer, which further helps to reduce dimensionality. ResNet uses the identity function (by directly connecting input to output) to preserve the gradient during backpropagation (as the gradient will be 1).

__DenseNet__

DenseNet, or dense networks, introduced the idea of connecting every convolutional layer with every other layer within (called a dense block). And every dense block is connected to every other dense block in the overall DenseNet. A dense block is simply a module of two 3x3 densely connected convolutional layers. These dense connections ensure that every layer is receiving information from all of the preceding layers of the network. This ensures that there is a strong gradient flow from the last layer down to the very first layer.

One key difference between ResNet and DenseNet is also that, in ResNet, the input was added to the output using skip connections. But in the case of DenseNet, the preceding layers’ outputs are concatenated with the current layer’s output. And the concatenation happens in the depth dimension. To controll the size of output (resulting from concatenation), a special type of block called the transition block is devised for this network. Composed of a 1x1 convolutional layer followed by a 2x2 pooling layer, this block standardizes or resets the size of the depth dimension so that the output of this block can then be fed to the subsequent dense block(s).

__Code: ResNet and DenseNet fine-tuned for binary classifications__ []()

## EfficientNets
EfficientNets is one of the best-performing CNN architectures. It use ts own optimization algorithm to search for the best scalling factor for the following three parameters: 1. model depth, 2. model width, and 3. model resolution. There three parameters used to manually scaling prior to EfficientNet.
1. model depth: though the timeline presented in the above, models add more layers to increase their power
2. model width: the number of feature maps or channels in a convolutional layer also has increased as more advanced architecutres emerge
3. model resoulution: going from 32x32 image in LeNet to 224x224 pixels in AlexNet (spatial dimension increase) also effect the model performance 

While increasing depth leads to increse model complexity, and increasing width leads to increase capability of model to lean more fine-grained features, there is a tradeoff. Deeper models suffer from vanishing gradient, and wider model suffer from a quick accuracy saturation. Lastly, higher resoulution data does not linearly equivalent of increase model performance. 

EfficientNet models find the architecture that has the right balance between depth, width, and resolution, concurrently scale all three parameters. It does so in two steps:
1. A basic architecture (called the base network) is devised by fixing the scaling factor to 1. At this stage, the relative importance of depth, width, and resolution is decided for the given task and dataset. The base network obtained is similar to MnasNet (Mobile Neural Architecture Search Network).
2. The optimal global scaling factor is then computed with the aim of maximizing the accuracy of the model and minimizing the number of computations (or flops).

The base network is called EfficientNet B0 and the subsequent networks derived for different optimal scaling factors are called EfficientNet B1-B7.

## Other architectures
- MobileNets: the goal is to retain peformance while reducing the model size
- CapsuleNet: revamped the convolutional units to cater to the third dimension (depth) in images.

## Task specific models

__Object detection and segmentaton__
- R-CNN
- Fast R-CNN
- Faster R-CNN
- Mask R-CNN
- Keypoint-RCNN
- ...

__Video-related tasks__
- ResNet3D
- ResNet Mixed Convolution:

---

# CNN + LSTM



