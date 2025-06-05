# New advancement in computer vision: overview

Since 2020, the field of computer vision has undergone transformative changes, largely driven by the introduction of transformer-based architectures and the consolidation efforts of projects like TIMM, which provide a unified reference for the community. These developments have spurred innovative approaches and steady advancements. While individual improvements may seem incremental and not headline-grabbing, their cumulative impact has been profound. In this repository, I compile a collection of these incremental enhancements that I have found valuable for modeling computer vision projects.

--- 

__Table of contents__
- ConvNeXt backbone architecture

# Swin Transformer

Swin Tansformer follows ConvNets approach using multi-stage design, in which each stage has different feature map resolution. Two key design consideration in the Swin Transformer model are:
1. stage compute ratio
2. stem cell structure

## Stage compute ratio
Changing the stage computation ratio of the ResNet-50 (change number of block in each stage from (3, 4, 6, 3) to (3, 3, 9, 3)), improves its accuracy from 78.8% to 79.4%. This is the concept that is being ussed in Swin Transoformer (1:1:3:1 in small version and 1:1:9:1 in large version).

---
# ConvNeXt

The ConvNeXt is a backbone architecture was introduced in 2022 by facebook. The core idea behind ConvNeXt was to re-examine the design choices of traditional CNNs (like ResNet) and "modernize" them by incorporating design elements that have proven successful in Vision Transformers (ViTs), without actually becoming a Transformer.

Changes result in the ConvNeXT backbone architecture could be divided into two main categories: macro and micro. Next section provides a summary of improvement at each level.
## Macro level (high-level structure)

###  and takeaways from architecture

## Micro level (block-level structure)
