# Lorenz-Equations-with-NN
An exploration in predicting future values of the Lorenz equations using various neural net machine learning architectures. 

##### Author: 
Riley Estes

### Abstract


### Introduction and Overview
The program aims to train neural networks in order to predict the solution to the Lorenz equations one step ahead of the current point. The Lorenz equations are notorious for being computationally unsolvable and only estimatable by calculating one point after another with mathematical techniques. Instead of using these techniques, we can instead use a neural network. To do so, 4 different neural network architectures will be tested for this application and the mean-squared errors of each compared. They are: Feed-Forward, Long Short-Term Memory, Recurrent Neural Network, and Echo State Network. 

### Theoretical Background

#### Lorenz Equations
The Lorenz equations are a system of ordinary differential equations that describe the behavior of a simplified atmospheric convection model. The equations exhibit chaotic behavior, characterized by sensitivity to initial conditions and the emergence of complex, unpredictable patterns. They have been widely studied in mathematics, physics, and other fields as a fundamental example of chaotic systems. They are known fer being unsolvable and only estimable using mathematical approximation techniques. In this context, they provide a complicated, sequential system that poses a challenge to predict using neural networks. 

#### Neural Network
A Neural Network, also known as a Multi-Layer Perceptron (MLP) is a machine learning algorithm where data passes through a series of layers of nodes connected with each other (fully or partially) by weights. This creates a nonlinear and very complicated network because each node in a layer is generally connected to all the nodes in the next layer, each with its own weight. That means that each node in a layer is the sum of all of the nodes in the last layer multiplied by each connection's particular weight. These networks require training with a training set, and are then tested on a test set of data. In training, the values of all the weights are updated (using backpropagation) based on the incoming data (and its labels for supervised learning). The model can then be tested on the test data to see how well it processed the training data, and how well its weights are set to achieve the data processing task. Neural Networks often perform very well on complicated tasks, but require huge amounts of data to do so. 

#### Feed-Forward Neural Network
A Feed-Forward Neural Network is one that has a linear flow of data from the input to the output. That is, there are no loops or ways data can be repeated or looped in the network. This is the simplist neural network design.

#### Long Short-Term Memory Neural Network (LSTM)
An LSTM is a type of Neural Network that is designed to process sequential data. Similar to a Recurrent Neural Network, an LSTM creates feedback loops so that it can "remember" data and use previous data in order to process current data. In addition to this however, the LSTM implements a memory cell where it can selectively store and access data in these cells for later use when processing future information. It adds an extra layer of memory to the Recurrent Neural Network design to further increase its temporal processing abilities. 

#### Recurrent Neural Network (RMM)
An RMM is a type of neural network designed to process sequential/time dependent data. Unlike feedforward neural networks that process data in a single forward pass, an RNN introduces the concept of "recurrence" by allowing information to persist and be passed from one step to the next. This enables the network to maintain an internal memory or state that captures the context and temporal dependencies of the sequential data. This allows the network to notice time-based patterns. 

#### Echo State Network (ESN)
An ESN (also known as a reservoir computing system) is a type of RNN where the recurrent connections within the network form a randomly initialized and fixed "reservoir" of neurons. The random initalization of the fixed reservoir weights ensures complexity in the model. These reservoir neurons have recurrent connections among themselves, creating a dynamic system capable of storing and processing information over time. Uniquely, only the connections from the reservoir to the output layer are learned, while the connections within the reservoir itself remain fixed. This means that during training, only the weights from the reservoir to the output layer are adjusted to learn the desired mapping or prediction task. 

### Algorithm Implementation and Development

### Computational Results

### Summary and Conclusions



