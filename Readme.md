# Simple Neural Network Implementation in C\#

This document provides a detailed breakdown of the provided C\# code, explaining its structure, functionality, and relationship to fundamental neural network concepts.

## Why This Code?

This code demonstrates a minimal **Feedforward Neural Network** with one hidden layer. Its primary goal is educational: to show how a neural network can be implemented from scratch in C\# and trained using a core machine learning principle: **Gradient Descent** (specifically, learning the function of addition).

## How the Code Works: Step-by-Step

### 1. Data Setup

| Variable | Structure | Example | Description |
| :--- | :--- | :--- | :--- |
| `Inputs` | Four samples, 2 features each. | $\{{1, 2\}, \{2, 3\}, \ldots\}$ | The values fed into the input layer. |
| `Outputs` | Four corresponding sums. | $\{3, 5, \ldots\}$ | The target values the network must learn to predict. |

### 2. Model Architecture 

* **Input Layer:** **2 neurons** (for the two input features, $x_1$ and $x_2$).
* **Hidden Layer:** **3 neurons**. This layer introduces non-linearity via the $\text{ReLU}$ function.
* **Output Layer:** **1 neuron** (for the final predicted sum, $\hat{y}$).

### 3. Initialization and Training

* **Parameters:** All initial weights ($W_1$ for Input→Hidden, $W_2$ for Hidden→Output) and biases ($B_1$, $B_2$) are set to **random values**.
* **Epochs:** The network is trained for **2000 iterations**.
* **Learning Algorithm:** **Stochastic Gradient Descent (SGD)** is used. The parameters are updated after processing *each* individual training sample.

### 4. The Training Loop (Per Sample)

| Step | Functionality | Calculation/Concept |
| :--- | :--- | :--- |
| **Forward Pass** | Calculates the prediction ($\hat{y}$) using current parameters. | $Z_1 = W_1 \cdot X + B_1 \rightarrow H = \text{ReLU}(Z_1) \rightarrow \hat{y} = W_2 \cdot H + B_2$ |
| **Error** | Determines the distance from the truth. | $Error = \hat{y} - y$ |
| **Weights Update** | Adjusts the parameters to reduce future error. | **$W_{new} = W_{old} - \eta \cdot \text{Gradient}$** |
| **Loss** | Accumulates the performance measure. | $\text{Squared Error} = (\hat{y} - y)^2$ |

> **Critical Note on Training:**
> The code only updates the **output layer weights ($W_2$) and bias ($B_2$)**. The input→hidden parameters ($W_1, B_1$) are **not** updated. This simplifies the implementation by avoiding the **backpropagation** step required to calculate gradients for the first layer.

## Functions Introduction

| Function | Formula / Logic | Purpose |
| :--- | :--- | :--- |
| `ReLU(x)` | $\max(0, x)$ | The **Activation Function** for the hidden layer. It introduces non-linearity, allowing the network to learn complex patterns. |
| `Predict(input)` | $W \cdot X + B$ (Linear) | Executes the **Forward Pass** through the two-layer network to generate an output. |
| `Loss(predicted, actual)` | $(\text{predicted} - \text{actual})^2$ | Implements the **Squared Error (L2 Loss)**, the cost function that the training process aims to minimize. |

## Neural Network & Training Summary

* **Network Type:** **Feedforward Neural Network** (specifically, a Multilayer Perceptron with a single hidden layer).
* **Activation:** The hidden layer uses **ReLU** (Rectified Linear Unit), and the output layer uses a **Linear** activation.
* **Learning Algorithm:** **Stochastic Gradient Descent (SGD)**.
* **Training Goal:** Minimize the **Mean Squared Error (MSE)** by iteratively adjusting weights and bias.

This model is a successful demonstration of the core forward-pass calculation and the gradient descent update mechanism, which are universal across nearly all modern neural network frameworks.
