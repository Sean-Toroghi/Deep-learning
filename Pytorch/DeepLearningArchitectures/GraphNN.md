<h1>Graph neural network</h1>

# GNN tasks
I start this summary with the tasks that be performed by graph neural networks. Later, I delve into the core principle and detail architecture.

Graph learning tasks can be divided into three main categories:
1. node-level tasks
2. edge level tasks
3. graph level tasks

## Node level tasks
The goal of node level tasks in to predict the class of a given node in the given graph. For example, it could be predicting the gender of each node. the latent feature representations of each node are then used to train a downstream task. This downstream task can be 1. a classification, 2. a regression, or 3. a clustering task. In this case, if graph represents a sentence, the POS tagging would be a node-level task, where each node( word) is assigned a POS tag.


## Edge level tasks
The goal of an edge level task is to classify edges. Each edge can be represented by a number derived from combining or concatenating the node-level features of the nodes at the two end of the edge. Also, edges could carry feature assigned to each edge. In most cases, the edge-level takes are used as accompanying of the node features to train a downstream classification. For example, the downstream task could be predicting the  dupe of relationship between nodes. If nodes represent people, the task would be to classify the relationship between connected people such as parent, sibling, and friend. In th field of computer vision, a GNN can be used to perform image scene understanding. In this example, the GNN predicts the relationship between objects in the image. This is equivalent of image scene understanding task.

## Graph level tasks
The goal of a graph level task could be to classify or regression done for the entire graph, sung the aggregation the latent features of all nodes in the graph. An example of this task is predicting the molecule type, in which each molecule contains several molecules in it. Another example is image classification , in which each pixel represents a node in the image. 


# GNN models
There are many GNN models proposed over the past decade, among which the more prominent ones are: GCN, GAT, and GraphSAGE. 

## GCN [^1]
GCN hs two blocks. The first block produce a vector for a set of nodes. The second block aggregates the output of the first block and generates the final output. The aggregation function for both blocks is averaging. GCN in summary uses averaging of information from neighbors to generate the final output. This help the model maintain the correct size for aggregation, but pose a limitation with the assumption neighbors are equality important.


## GAT [^2]

GAT employs attention mechanism for aggregation among the output of different neighbors. It introduces a new set of trainable pramaters (weight for each neighbor, represented by the attention mechanism) in the form of attention vector. It is twice the length of an individual node feature vector to accommodate dot-multiplicated by the concatenation of a given nodde's feature with its negibor at a time. At a given layer, the attention vector is shared among all (node. neighbor) pairs. This way the model can learn different weights for different feature dimensions and also for different neighbors. 

One downside of GAT is scalability, in which th size of the graph is a bottleneck for the model. 

## GraphSAGE[^3]

Graph sample and aggregate (GraphSAGE) randomly and uniformly samples neighbors of a given node, and uses only those selected ones to extract graph information, instead of using all neighbors. This help the model handling large graph dataset. The author of the original GraphSAGE paper suggest three aggregation methods: average, LSTM, and MaxPool. 

## PinSAGE[^4]
PinSAGE is an extension of GraphSAGE developed and is used by Pinterest as a part of their recommendation system algorithm. Their model handles more than 3 Billion nodes (users) and 18 Billion edges. 


















---
<mark>Status</mark>
- [x] Summary
- [ ] Project added
- [ ] Finalized
---

[^1]: GCNs: https://tkipf.github.io/graph-convolutional-networks/
[^2]: GANs: https://arxiv.org/abs/1710.10903
[^3]: Inductive Representation Learning on Large Graphs: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
[^4]: Graph Convolutional Neural Networks for Web-Scale Recommender Systems: https://arxiv.org/abs/1806.01973






