# Basic Machine Learning Setup in C\# Console App

This document provides a detailed explanation of a single-neuron linear regression model implemented from scratch in a C\# Console Application.

## Model Overview

| Category | Detail |
| :--- | :--- |
| **Model Type** | **Linear Regression** |
| **Learning Algorithm** | **Stochastic Gradient Descent (SGD)** |
| **Architecture** | **Single Neuron** (Perceptron without an Activation Function) |
| **Function** | Linear mapping: $\hat{y} = W_1 x_1 + W_2 x_2 + W_3 x_3 + B$  |
| **Goal** | To learn the weights ($W$) and bias ($B$) that minimize the prediction error on the training data. |

## What the Code Accomplishes

The code demonstrates the fundamental machine learning process:

1.  **Training:** Trains a linear model to fit a set of input-output pairs by learning the optimal weights and bias.
2.  **Optimization:** Uses **Gradient Descent** to iteratively adjust these parameters to reduce the prediction error (specifically, the Mean Squared Error).
3.  **Prediction:** After training, the model uses the learned parameters to make predictions on new, previously unseen input values.

## Code Explanation: Step-by-Step Breakdown

### 1. Data Setup

| Element | Detail | Purpose |
| :--- | :--- | :--- |
| `inputs` | 2D array of 9 samples, 3 features each (e.g., $\{1.0, 2.0, 1.0\}$). | Provides the independent variables used to train the model. |
| `outputs` | 1D array of 9 target values. | Provides the dependent (true) values the model must learn to predict. |

### 2. Model Initialization

* **`weights` (3 values):** Initialized randomly. Each weight corresponds to one input feature ($x_1, x_2, x_3$).
* **`bias` (1 value):** Initialized randomly.

> **Why Random?** Random initialization prevents the **symmetry problem**, ensuring that all three weights start with unique values and can learn different contributions during training.

### 3. Training Loop (Stochastic Gradient Descent)

The training is controlled by two nested loops:

* **Outer Loop (`epochs = 1000`):** Repeats the learning process 1000 times over the entire dataset.
* **Inner Loop (Per Sample):** Iterates through each of the 9 training samples.

| Training Step | Formula / Concept | Explanation |
| :--- | :--- | :--- |
| **Predict** | $\hat{y} = W \cdot X + B$ | Performs the **Forward Pass**: Calculates the current output based on the input and the model's current parameters. |
| **Error** | $Error = \hat{y} - y$ | Calculates the residual, which is the direct measure of how wrong the current prediction is. |
| **Weights Update** | $W_{new} = W_{old} - \eta \cdot \text{Error} \cdot x_i$ | **Gradient Descent:** Adjusts the weight by moving in the direction that decreases the loss, scaled by the `learningRate` ($\eta$). |
| **Bias Update** | $B_{new} = B_{old} - \eta \cdot \text{Error}$ | Adjusts the model's vertical offset based on the average error. |
| **Loss** | $\text{totalLoss} += (\hat{y} - y)^2$ | Accumulates the **Squared Error** to monitor the model's performance over the epoch. |

### 4. Prediction

* After the 1000 epochs, the final learned weights and bias are used to predict outputs for two new input vectors: $\{5, 5, 5\}$ and $\{20, 5, 6\}$.
* The model generalizes the learned relationship (the weighted sum) to make predictions on **unseen data**.

### 5. Helper Functions

| Function | Core Calculation | Role |
| :--- | :--- | :--- |
| `Predict` | Weighted Sum + Bias | The mathematical core of the model; computes the linear relationship. |
| `Loss` | Squared Error | The **Cost Function**; measures the penalty for poor predictions.  |
