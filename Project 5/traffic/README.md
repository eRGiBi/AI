
First I tried the same neural network that was showed in the lecture,
but got a 0.0542 accuracy as a result.

I set MaxPooling to a size of 2 by 2 matrix, 
a single hidden layer of 128 neurons, with the activation function of relu, 
set the dropout to 0.5,
and the output lay's activation function to softmax.

First because of the worsening accuracy 
I thought I'd set the dropout to higher  lower but this made the results even worse.
And then lower which sligthly improved.

Adding another hidden layer with 128 or 8 neuron did'nt improve, 
nor did adding more than 1 layer. 

A 3 by 3 pool size no

mean_squared_error
 tanh

With the setup of a NN with 1 Conv, 1 2D MaxPool, 3 Dense (728, 512, 256) unit, 
using adam, BinnaryCrossentropy, I found that the model reached almost
95% accuracy in 30 epochs and 100% in 50, but an interesting overfitting occurred,
with exactly 0.6667 accuracy on the validation data.

Increasing the dropout lessened both the training and validation accuracy.

