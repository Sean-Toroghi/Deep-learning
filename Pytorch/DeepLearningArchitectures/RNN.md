# Deep learning with Pytorch: Recurrent neural network



__Codes__
- [Train RNNs for sentiment analysis with PyTorch]()
 

## Recurrent neural networks
Recurrent neural networks are designed for handling sequence datasets, in which datapoints are temporal (each data point has a dependency to some of the other datapoints in the series). The dependency could be among inputs (x1, x2, ..., xt) or/and outputs (y1,y2,...,yt). RNNs are capable of modeling the temporal aspect of data by including additional weights in the model that create cycles in the network. The architecture uses recurrent connections across time steps. A recurrent connection is an intermediate outputs at time step _t_ as an input for the next time step _t+1_, while the model maintains hidden internal state. One of the powerful features of RNNs is that they can deal with sequential data of varying sequence lengths (T). Thre are several recurrent neural network architectures, among which are LSTM, and GRUs. 

## Recurrent models and input-output relation
Recurrent neural networks can model all types of input-output relationships:
- many-to-many: such as machine translation(encoder-decoder) or NER (instantaneous)
- one-to-many: such as image captioning
- many-to-one: such as sentiment analysis
- one-to-one: such as image classification by processing image pixel sequentially

## History of RNNs
While the introduction of RNNs goes back to 1986, it goes a long way to reach the current state. Some of key checkpoints in time are the introduction of bidirectional RNN and LSTM in 1997, bidirectional LSTM in 2003, stacked LSTM in 2013, GRU in 2014, grid-LSTM in 2015, and gated orthogonal recurrent units in 2017. 

Bidirectional RNN improves the efficiency of a RNN model and LSTM addressed issue of exploding and vanishing gradient. LSTM uses a series of cell states, and gates to control the flow of information to the next cell while preserving or forgetting the information that comes from the previous cell. Stacked LSTM improves model capability to learn more complex patterns across various sequential processing tasks such as speech recognition. GRU architecture simplified LSTM by using less state and gates, and grid-SLTM is the LSTM equivalent of multidimensional RNNs. In grid-LSTM the cells are arranged into a multi-dimensional grid, and are connected along the spatiotemporal dimensions of the data, as well as between the network layers. Gated orthogonal recurrent units combined the idea of GRUs and unity RNNs (using unitary matrices (which are orthogonal matrices) as the hidden-state loop matrices of RNNs to deal with the problem of exploding and vanishing gradients). 
