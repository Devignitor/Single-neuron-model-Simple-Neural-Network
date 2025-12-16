Very Basic Tiny  Machine Learning setup in C# Console App
Model Type: Linear Regression
Learning Algorithm: Stochastic Gradient Descent (SGD)

Model is a single neron based linear regression model with 3 inputs and 1 output.
it works without activation function.
	y^​=W1​x1​+W2​x2​+B
It is the most fundamental simple neural network.


The purpose of this code is to demonstrate a simple implementation of linear regression using gradient descent in C#.
What the code does:
	Trains a linear model to fit a set of input-output pairs (training data).
	Learns weights and a bias so that the model can predict an output value given a new set of three input values.
	Uses gradient descent to iteratively adjust the weights and bias to minimize the prediction error (mean squared error).
	After training, the model predicts outputs for new, unseen input values and prints the results.

This code is a basic example of machine learning, specifically linear regression, showing how a computer can "learn" the relationship between inputs and outputs from data and then make predictions for new data.


Code Explaination
1. Data Setup
	inputs: A 2D array of 9 training samples, each with 3 features (e.g., {1.0, 2.0, 1.0}).
	outputs: A 1D array of 9 target values, each corresponding to a row in inputs.
2. Model Initialization
	weights: An array of 3 weights (one per input feature), initialized randomly.
	bias: A single bias value, also initialized randomly.
3. Training Loop
	Epochs: The model trains for 1000 epochs (iterations).
	For each epoch:
	totalLoss: Accumulates the squared error for all samples in the epoch.
	For each training sample:
	Predict: Calculates the model’s output as a weighted sum of inputs plus bias.
	Error: The difference between the predicted value and the actual output.
	Weights Update: Each weight is adjusted by subtracting a fraction of the error, scaled by the input and learning rate.
	Bias Update: The bias is adjusted by subtracting a fraction of the error, scaled by the learning rate.
	Loss Calculation: The squared error is added to totalLoss.
	Every 100 epochs, the current loss is printed.
4. Prediction
	After training, the model predicts outputs for two test inputs:
	{5, 5, 5}
	{20, 5, 6}
	The predictions are printed to the console.
5. Helper Functions
	Predict: Computes the weighted sum of inputs plus bias.
	Loss: Computes the squared error between predicted and actual values.