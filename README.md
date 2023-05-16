# Lorenz-Equations-with-NN
An exploration in predicting future values of the Lorenz equations using various neural net machine learning architectures. 

##### Author: 
Riley Estes

### Abstract
A simple Feed-Forward Neural Network, Long Short-Term Neural Network, Recurrent Neural network, and Echo State Network are all standardized to the same 3 layer shape with the same hyperparameters and tested on their abilities to predict the solutions to the Lorenz equations when the value of rho in the equations changes. The feed-forward neural network performs the best by far, with the RNN coming in second place, and the other two methods performing very poorly. Seeing as the two best solutions are also the simplist in terms of model complexity, it would seem that solving the Lorenz equations are not as complicated as one would seem, or that the more complicated models are more prone to "overthinking" the solution. 

### Introduction and Overview
The program aims to train neural networks in order to predict the solution to the Lorenz equations one step ahead of the current point. The Lorenz equations are notorious for being computationally unsolvable and only estimatable by calculating one point after another with mathematical techniques. Instead of using these techniques, we can instead use a neural network. To do so, 4 different neural network architectures will be tested for this application and the mean-squared errors of each compared. They are: Feed-Forward, Long Short-Term Memory, Recurrent Neural Network, and Echo State Network. 

### Theoretical Background

#### Lorenz Equations
The Lorenz equations are a system of ordinary differential equations that describe the behavior of a simplified atmospheric convection model. The equations exhibit chaotic behavior, characterized by sensitivity to initial conditions and the emergence of complex, unpredictable patterns. They have been widely studied in mathematics, physics, and other fields as a fundamental example of chaotic systems. They are known fer being unsolvable and only estimable using mathematical approximation techniques. In this context, they provide a complicated, sequential system that poses a challenge to predict using neural networks. An example of 100 random points and their solution lines over time is given here: 
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/GT28.png" width="400"/>

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
Firstly, some parameters for calculating the Lorenz equations and defining one step in time are initialized:
```
dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10
rho = 28
```
Then the Lorenz derivative equations are defined in the function:
```
def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
  x, y, z = x_y_z
  return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
```
generate_data is defined to generate the solved pathways in x_t that the initial points in x0 take according to teh Lorenz equations. integrate.odeint makes these calculations for 100 random points:
```
x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(sigma, beta, rho)) for x0_j in x0])
```
nn_input is populated as each point in the generated pathways in dt long time intervals. nn_output is the nn_input array at t + 1, (or t + dt) which will act as a labels/ground truth array for the neural network training when given the nn_input array to train on. 

The LorenzModel feed-forward neural network is defined with the following layers connected with ReLu activations:
```
self.fc1 = nn.Linear(6, 128)
self.fc2 = nn.Linear(128, 128)
self.fc3 = nn.Linear(128, 3)
```
One hidden layer of size 128 will be the standard shape for all NN architectures tested here in order to get a baseline for how the models themselves effect the loss instead of hyperparameter tuning. 
Additionally for all models, the Adam optimizer is used with 0.001 learning rate and MSE loss:
```
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
The model is trained on 3 different values of rho in the Lorenz equations being 10, 28, and 40. Note that the model weights carry over between each value of rho, so it consistently trains the model for different rho values in order to generate an estimate for different values of rho. 
```
for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(nn_input_tensor)
    loss = criterion(outputs, nn_output_tensor)
    loss.backward()
    optimizer.step()
```
All models in this program are trained in a very similar way for 30 epochs per value of rho. 

Then, for rho values 17 and 35, each model is tested in a similar way to the feed forward here:
```
predictions = model(nn_input_tensor)
loss = criterion(predictions, nn_output_tensor)
```
Each other model is initialized and trained very similar to the feed-forward. Note that the ESN must also initialize and manage a reservoir as well. 

After the trainings, the models are tested on rho = 17 and 35 and the results are graphed. 

### Computational Results
#### Feed-Forward NN
The results for the feed-forward NN were very good. For rho = 10, the loss (MSE) started at 47 and went down to 1.5. For rho = 28, it started at 32 and ended at 2.1. For rho = 40, it started at 3.2 and ended at 0.23. Overall the model improved greatly as it trained across different values of rho. 
When evaluated on rho = 17, the loss was 2.9 and the model plotted this:
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/NNPreds17.png" width="400"/>
Compared to the ground truth generated by integrate.odeint:
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/GT17.png" width="400"/>
For rho = 35, the feed-forward NN had a loss of only 0.3690 and generated this plot:
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/NNPreds35.png" width="400"/>
To be compared against the ground truth for rho = 35:
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/35GT.png" width="400"/>
Note that in the predictions, the lines don't connect to the same random starting points because the model's first prediction is t + dt, so where the point is after dt time has passed. 
The model only took 16 seconds to train as well, making these results very efficient. Although the loss would increase after changing the value of rho, the model could quickly and effectively readjust for it, and by the third value of rho, had already become very accurate and adaptable. This was seen in the test loss as well. The model is slightly biased towards the training data it saw last, which explains why the test loss on rho = 17 is higher than the loss for rho = 35. For the time this model took to train, the results are superb, and proves to be very adaptable to different values of rho and changing datasets. 

When training the other models, the results are as follows:

#### LSTM
The LSTM did not perform quite as well as the feed-forward:
for rho = 10:
start loss: 43
end loss: 12

for rho = 28:
start loss: 188
end loss: 114

for rho = 40:
start loss: 300
end loss: 219

and tested:
for rho = 17
loss = 10.29

for rho = 35
loss = 144.2

and produced the following graphs:
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/LSTM17.png" width="400"/>
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/LSTM35.png" width="400"/>

and took 4 minutes to train.

The LSTM performed very poorly after the first rho value, and even then did not fit too well to rho = 10. It was able to generate an okay amount of loss when tested on rho = 17, but still fell short to the feed-forward algorithm. This is likely a sign of overtraining to rho = 10, and not being able to train further past that point. With standardized hyperparameters, this can happen if the parameters are not chosen well for the algorithm or if the model relies more heavily on hyperparameter tuning. It should be noted however that the FF NN did not require any tuning at all and worked right from the start very quickly. The graphs also look nothing like they should and instead converge around one or two points for both rho = 35 and rho = 17, showing terrible adaptability to new rho values. Seeing as this model took longer to train and still performed worse than the FF NN across all rho values but also when trained only on rho = 10, I would not recommend the model unless great hyperparameter tuning could outperform the feed-forward with the same amount of tuning. 

#### RNN
The RNN performed only slightly better than the LSTM:
for rho = 10:
start loss: 3391
end loss: 30

for rho = 28:
start loss: 1172
end loss: 108

for rho = 40:
start loss: 554
end loss: 166

and tested:
for rho = 17
loss = 76.9

for rho = 35
loss = 1.92

and produced the following graphs:
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/RNN17.png" width="400"/>
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/RNN35.png" width="400"/>

and took a whopping 23.5 minutes to train. 

The RNN did not perform well but shows promise of being able to adapt to new data sets slowly but consistently. It is however produce the only non FF NN graphs that look similar to how they should, and seems to have somewhat accurately predicted the right shape the data takes even if the exact values are off. This is a good sign of potential. The starting errors are very high, but reduced as new rho values were introduced, and there are large gaps between where the error started and where it ended for each rho value during training. Also, the RNN made the best test predictions for a single value of rho out of the non FF NN architectures with only 1.92 loss for rho = 35. Due to its high performance on only rho = 35, it would seem that this model overtrained to the last value of rho it trained to, being something worth taking into consideration, but also proving how adaptable and willing to change it is for new data. The FF NN still outperformed the RNN with much greater efficiency, but the RNN may have a practical application to make accurate predictions on newer data with hyperparameter tuning and a lot of processing power and time, and shows a good ability to get the general shape of the data correct which is a sign that it is predicting the output although poorly. 

#### ESN
Finally, the ESN performed about the same as the RNN but was more specialized for rho = 10:
for rho = 10:
start loss: 46
end loss: 11.6

for rho = 28:
start loss: 207
end loss: 121

for rho = 40:
start loss: 337
end loss: 233

and tested:
for rho = 17
loss = 14.9

for rho = 35
loss = 162.3

and produced the following graphs:
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/ESN17.png" width="400"/>
<br> <img src="https://github.com/rileywe/Lorenz-Equations-with-NN/blob/main/LorenzOutputImgs/ESN35.png" width="400"/>

and took 11.2 seconds to train.

The ESN was the fastest model to train but also suffered heavily from changing rho values in the training set much like the LSTM did. This model overall performed very similar to the LSTM, and both seem to have overtrained to the first value of rho (hence the higher test accuracy when rho = 17) and showed a defiance to adapting to different values of rho. Much like the LSTM, the FF NN was able to outperform it in nearly every way without overtraining. Also, the graphs seemed to converge on two points like the LSTM graphs did for both test rho values, which is a sign of terrible adaptability to different rho values. It's interesting however that both the ESN and LSTM had roughly the same graph behavior, both characteristic of overtraining and non-adaptability. In both cases, any good error score may likely be due to luck rather than accurate predictions. This makes sense because the ground truth graph for rho = 17 also converges near where the LSTM and ESM graphs do. I would not recommend this method unless hyperparameter tuning could greatly improve the model. It does train faster than the other models tried here, so this has the potential to create results for cheap. 

### Summary and Conclusions
Overall, all of these models except for the feed-forward neural network overtrained, didn't train fast enough, and could not adapt to predict multiple values of rho. The feed-forward neural network on the other hand was able to adapt to new datasets with different values of rho, and test well on those it hadn't seen before. Due to the simplicity of the model, it's an obvious first choice for solving the Lorenz equations or for other similar applications because it works so well for being quick and cheap. As for why the simplist model performed the best, perhaps it's because the other models are looking for patterns that simply aren't there. The Lorenz equations are used in the study of chaos theory, so it's likely that the simpler neural network regression solution solves the problem in the simplist way without overthinking it and trying to fully understand the chaotic interactions in the equations. The solutions also gravitate to a disc shape like a 2d plane, so maybe the solution is best found with a simple regression solution that aims to find that plane. Notice too how the second least complicated model, the RNN, performed second best. Though these are the observed results with semi-randomly chosen hyperparameters, it should be noted that none of these models had any hyperparameter tuning. Perhaps the models that failed here could do well given enough attention and engineering, and a few showed promise and adaptable behavior that could be built upon. In conclusion, the feed-forward neural network is shown here to be the best choice for predicting the next point in time in the Lorenz equations solution. 


