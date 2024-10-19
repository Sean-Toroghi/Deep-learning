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


## the advantages of CNN
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
LeNet is one of the earliest CNN models, consists of 5 layers and 60k parameters. It has two block of CNN and avg pooling after each block. The there are two fully connected blocks that generate an output array is size 10 (number of target classes).

 __Code:__ implementation of LeNet in Pytorch []()

 ## AlexNet
 AlexNet is the successor of LeNet, 

















