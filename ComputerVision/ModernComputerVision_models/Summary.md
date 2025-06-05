# New advancement in computer vision: overview

Since 2020, the field of computer vision has undergone transformative changes, largely driven by the introduction of transformer-based architectures and the consolidation efforts of projects like TIMM, which provide a unified reference for the community. These developments have spurred innovative approaches and steady advancements. While individual improvements may seem incremental and not headline-grabbing, their cumulative impact has been profound. In this repository, I compile a collection of these incremental enhancements that I have found valuable for modeling computer vision projects.

--- 

## Table of contents
- Swin-Transformr
- ConvNeXt backbone architecture


---
---

# Swin Transformer

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

# ConvNeXt

The ConvNeXt is a backbone architecture was introduced in 2022 by facebook. The core idea behind ConvNeXt was to re-examine the design choices of traditional CNNs (like ResNet) and "modernize" them by incorporating design elements that have proven successful in Vision Transformers (ViTs) and Swin-Transformer, without actually becoming a Transformer.

Changes result in the ConvNeXT backbone architecture could be divided into two main categories: macro and micro. Next section provides a summary of improvement at each level.
- Macro level (high-level structure)
- Micro level (block-level structure)

## Development of ConvNeXt
To develop the ConvNeXt architecture, authors started from ResNeXt model.
