# Advancement in computer vision: overview

Since 2020, the field of computer vision has undergone transformative changes, largely driven by the introduction of transformer-based architectures and the consolidation efforts of projects like TIMM, which provide a unified reference for the community. These developments have spurred innovative approaches and steady advancements. While individual improvements may seem incremental and not headline-grabbing, their cumulative impact has been profound. In this repository, I compile a collection of these incremental enhancements that I have found valuable for modeling computer vision projects.

Three factors effect the performance of a visual representation learning system, in general:
1. model architecture
2. training method
3. available data used for training

---  

## <a name="table">Table of contents</a>
- [Swin-Transformr](#swin)
- [ConvNeXt backbone architecture](#convnext)
- [ConvNeXt V2](#connext2)
- [MetaFormer for Vision](#metaformer)
- [EVA](#eva)


---
---

# <a name="swin">[Swin Transformer](#table)</a>

Swin Tansformer follows ConvNets approach using multi-stage design, in which each stage has different feature map resolution. Two key design consideration in the Swin Transformer model are:
1. stage compute ratio
2. stem cell structure

## Stage compute ratio
Changing the stage computation ratio of the ResNet-50 (change number of block in each stage from (3, 4, 6, 3) to (3, 3, 9, 3)), improves its accuracy from 78.8% to 79.4%. This is the concept that is being used in Swin Transoformer (1:1:3:1 in small version and 1:1:9:1 in large version). 

Perhaps more optimal ratio is likely to exist, as there has been ongoing research about the optimal distribution of computation. Check the following paper for more information: 
- Designing Network Design Spaces (2020) [arxiv](https://arxiv.org/pdf/2003.13678)

## Stem cell structure

The first layer of a model reduces the size of an input image. This is an efficient approach, since there are the inherent redundancy in natural images. The ViT model architecture a _patchify_ strragy is employed as stem cell (with large kernel size and non-overlapping conv). Swin-Transformer employs the same _patchify_ strategy with much smaller kernel size (4x4). The replacement of Res-Net stem cell with a 4x4 kernel and stride 4 convolutional layer increases the accuracy from 79.4% to 79.5%. 

__Note__: 
- Vision Transformers (ViTs) process entire images as patches and apply self-attention globally, meaning every token (patch) interacts with all others. This is a downside of ViT, as it creates a quadratic complexy.
- Swin Transformer divides the image into smaller windows and applies self-attention inside each window separately. This change makes _patchify_ approach applicable for large images as it recudes the complexity of the model from quadratic $O(n^2)$ to linear $O(n)$.
- Furthermore, the Swin-Transormer maintains the information flow when transitioning from one windows to the next one by introducing the concept of _shifting windows_, which allows cross window connections.

---
---

# <a name="convnext">[ConvNeXt](#table)</a>

The ConvNeXt is a backbone architecture was introduced in 2022 by facebook. The core idea behind ConvNeXt was to re-examine the design choices of traditional CNNs (like ResNet) and "modernize" them by incorporating design elements that have proven successful in Vision Transformers (ViTs) and Swin-Transformer, without actually becoming a Transformer.

To develop the ConvNeXt architecture, authors started from ResNeXt model. Changes result in the ConvNeXT backbone architecture could be divided into two main categories: macro and micro. Next section provides a summary of improvement at each level.
- Macro level (high-level structure)
- Micro level (block-level structure)

---
## Macro-level development of ConvNeXt

---
## Micro level development of ConvNeXt

At the micro level, the changes have been made are at layer level. The summary of changes are as following:
- change activation from ReLU to GeLU
- redue number of activation fucntions
- reduce number of normalization layers
- substitude batch normalization with layer normalization
- seperating the downsampling layers

### change of activation function
While the original transformers model and ocnventional convolution models all use ReLU activatio function, modern models including BERT, GPT-2, and ViT replace the ReLU with Gaussian Error Linear Unit (GELU). GELU can be seen as a smoother version of ReLU. In the case of ConvNeXt, the replacement of ReLU to GELU does not result in any change in accuracy (maintain at 80.6%). 

### Reduce number of activation funtions
Comparing the activation function placement and cocunt in the Transformers and ResNet architectures shows that the Transformer architecure employ much less number of activation functions. In a Transfomers block there are two activation functions in each block (one after each of the two FFN layers). In the ResNET, after each convolution layer, there is an activation function. This mean each ResNet block has multiple activation function.

Mimicking the number of activation functions in Transformers, by eliminating all of the activation function except the one between two 1x1 layers increases the accuracy of the ConvNeXt by 0.7% point to 81.3%.

### Reduce the number of normalization layers

Transoformers compare with ResNet has fewer batch-normalization layers. Having a single BN layer (removing two BN layers) before the conv 1x1 layers increases the accuracy of the ConvNeXt to 81.4% (0.1% point increase). 

### Replacing BN layers with LN
While BN layers improves the convergence and reduces overfitting, it also has detrimental effect of the model's performance <sup>[1](#1)</sup>. While it has been reported substituting BN with LN results in suboptimal performance <sup>[2](#2)</sup>, this replacement in ConvNeXt not only does not negatively effect training, but also slightly improves accuracy to 81.5%. Also adding normalization layers wherever spatial  resolution is changed can help stablize training.

### Seperating the downsampling layers
Employ 2x2 conv layers with stride 2 for spatial downsampling (similar to Swin-Transformer) improves the accuracy to 82%.

## ConvNeXt variants
ConvNeXt model comes in five variants: T, S, B, L, and XL. The T and B versions are the upgrades/modernizations applies to ResNet-50 and ResNet-200 respectively. The variants are differ in number of blocks (B) and channels (C). A summary of configurations is shown below:-
- ConvNeXt-T: C = (96192384768), B = (3393)
- ConvNeXt-S: C = (96192384768), B = (33273)
- ConvNeXt-B: C = (1282565121024), B = (33273)
- ConvNeXt-L: C = (1923847681536), B = (33273)
- ConvNeXt-XL: C = (25651210242048), B = (33273



---
---

# <a name="convnext2">[ConvNeXt V2](#table)</a>

ConvNeXt V2 employs several features, including a fully convolutional masked autoencoder framework and a new Global Response Normalization (GRN) layer, added to the original ConvNext architecture. These additions enhances inter-channel feature competition, resulting in improvement of the performance in variety of tasks including image- classification, detection, and also segmentation.  

Furthermore, the second verison of the ConvNeXt model is compatible with masked autoencoders (MAE) technique (a self-suprevised training approach)  <sup>[4](#4)</sup>. A study by Jing et al. (2022) shows training a ConvNets with mask-based self-supervised learning can be difficult <sup>[3](#3)</sup>.

## ConvNeXt v2: fully convoluitonal masked autoencoder
This model employs a simple but effective self-supervised technique: geneate learning signal by heavily mask the input image (ratio of 0.6 ), and then the objective is for the model to  predict the masked areas, given the remaining context. This masking is applied to the last downsampled stage (32x32 patch), and then an upsampling is performed to reach the original image size. 

__Agmentation__: at this stage only augmentation that is used is random resized cropping.

The model architecture consists of two parts: __encoder and decoder__.

- ___Encoder__: for encoder the model employs ConvNeXt. To prevent the model from learning shortcuts (copy and paste information from masked regions), the authors converted the standard convolution layer in the encoder with the submanifold sparse convolution. This change enables the model to operate only on the visible data points.
- __Decoder__: the authors employ a single ConvNeXt block decoder, as a light version of ConvNeXt. This choice of decoder, instead of a hierarchical decoders or transformers, forms an asymmetric encoder-decoder architecture overall while deliver high fine-tunning accuracy and reduced pre-training time.
- __Reconstruction an error__: similar to MAE, the authors employ the mean squared error (MSE) between the reconstructed and target images. 

## Global response normalization (GRN)
To make the training more effective in conjunction with the ConvNeXt architecture, authors proposed a new technique: global response normalization. The original fully convolutional masked autoencode pre-trained ConvNeXt-Based model suffers from feature collapse (many dead or saturated feature maps leading to the activation becomes redundant across channels). To prevent feature collapse, we need to find a way to diversify the features during the learning phase.

To address the challenge of feature collapse, authors propose a new response normalization layer called global response normalization (GRN), which aims to increase the contrast and selectivity of channels. A GRN unit consists of three steps: 
1. global feature aggregation: L-2 norm
2. feature normalization: standard divisive normalization
3. feature calibration: calibrate the original input responses using  the computed feature normalization scores

  ```python
  def GRN(X):
    # gamma, beta: learnable affine transform parameters
    # X: input of shape (N,H,W,C)
    gx = torch.norm(X, p=2, dim=(1,2), keepdim=True)
    nx = gx / (gx.mean(dim=-1, keepdim=True)+1e-6)
    return gamma * (X * nx) + beta + X
  ```





---
---

# <a name="metaformer">[Paper (2022 - revised 2024) MetaFormer Baselines for Vision](#table)</a>


---
---
# <a name="eva">[Paper (2022) EVA: Exploring the Limits of Masked Visual Representation Learning at Scale
](#table)</a>










---
---
__References__
1. <a name="1">Yuxin Wu and Justin Johnson. Rethinking "batch" in batch norm. arXiv:2105.07576, 2021</a>.
2. <a name="2">Yuxin Wu and Kaiming He. Group normalization. In ECCV, 2018</a>.
3. <a name="3">Li Jing, Jiachen Zhu, and Yann LeCun. Masked siamese convnets. arXiv preprint arXiv:2206.07700, 2022</a>.
4. <a name="4">Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022</a>.
