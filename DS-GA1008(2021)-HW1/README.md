# Homework 1: Backpropagation
## Deep Learning (NYU) 2021
Link: https://atcold.github.io/NYU-DLSP21/
Following is my answers for the theory part of the homework, they are not verified to be correct. Take it with a big grain of salt.
### Theory
#### Instructions:
1. Every vector is treated as a column vector
2. Use [numerator-layout notation](https://en.wikipedia.org/wiki/Matrix_calculus#Numerator-layout_notation) for matrix calculus. 
3. Only use vector and matrix (no tensor)
4. Missing transpose are wrong.

#### 1.1 Two-Layer Neural Nets
We have the following neural net:  
$$\boldsymbol{x} → \text{Linear}_1 → f → \text{Linear}_2 → g → \boldsymbol{\hat{y}}, $$
where $\text{Linear}_i(x) = \boldsymbol{W^{(i)}}\boldsymbol{x} + \boldsymbol{b^{(i)}}$, and $f, g$ are element-wise nonlinear activation functions. $\boldsymbol{x} \in \mathbb{R}^n$, $\boldsymbol{\hat{y}} \in \mathbb{R}^K$.

#### 1.2 Regression Task
Choose $f = \text{ReLU}$, and $g$ to be an identity function. We choose MSE as the loss function: $l_{MSE}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = \Vert \boldsymbol{y} - \boldsymbol{\hat{y}} \Vert^2$
1. **Name and mathematically describe the 5 programming steps you would take to train this model with `PyTorch` using SGD on a single batch of data.**
* Step 1: set all the gradients to zeros as 
$$\frac{\partial l_{MSE}}{\partial\boldsymbol{W^{(i)}}} = 0, \frac{\partial l_{MSE}}{\partial\boldsymbol{b^{(i)}}} = 0$$
* Step 2: do the forward pass, put $\boldsymbol{x}$ through the network and get $\boldsymbol{\hat{y}}$
* Step 3: calculate the loss $l_{MSE}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = \Vert \boldsymbol{y} - \boldsymbol{\hat{y}} \Vert^2$
* Step 4: calculate the gradients of the loss function with respect to the weights $$\frac{\partial l_{MSE}}{\partial\boldsymbol{W^{(i)}}}, \frac{\partial l_{MSE}}{\partial\boldsymbol{b^{(i)}}}$$
* Step 5: update the weights according to the calculated gradients as
$$
\begin{align}
\boldsymbol{W^{(i)}} ← \boldsymbol{W^{(i)}}-\gamma\frac{\partial l_{MSE}}{\partial\boldsymbol{W^{(i)}}}, \\  \boldsymbol{b^{(i)}} ← \boldsymbol{b^{(i)}}-\gamma\frac{\partial l_{MSE}}{\partial\boldsymbol{b^{(i)}}}
\end{align}
$$
2. **For a single data point $(\boldsymbol{x}, \boldsymbol{y})$, write down all inputs and outputs for forward pass of each layer. You can only use variable $\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{W}^{(1)}, \boldsymbol{b}^{(1)}, \boldsymbol{W}^{(2)}, \boldsymbol{b}^{(2)}$ in your answer (note that $\text{Linear}_i(x) = \boldsymbol{W^{(i)}}\boldsymbol{x} + \boldsymbol{b^{(i)}}$).**

| Layer                 | Input                | Output        |
| :---                  |    :----:            |          ---: |
| $\text{Linear}_1$     | $\boldsymbol{x}$     |  $\boldsymbol{z}_1 = \boldsymbol{W^{(1)}}\boldsymbol{x} + \boldsymbol{b^{(1)}}$  |
| $f$                   | $\boldsymbol{z}_1$        | $\boldsymbol{z}_2 = \text{ReLU}(\boldsymbol{z}_1)$     |
| $\text{Linear}_2$     | $\boldsymbol{z}_2$      | $\boldsymbol{z}_3 = \boldsymbol{W^{(2)}}\boldsymbol{z}_2 + \boldsymbol{b^{(2)}}$   |
| $g$                   | $\boldsymbol{z}_3$        |   $\boldsymbol{\hat{y}} = \boldsymbol{z}_3$    |
| Loss                  | $\boldsymbol{\hat{y}}, \boldsymbol{y}$       | $\Vert \boldsymbol{y} - \boldsymbol{\hat{y}} \Vert^2$ |

3. **Write down the gradient calculated from the backward pass. You can only use the following variables: $\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{W}^{(1)}, \boldsymbol{b}^{(1)}, \boldsymbol{W}^{(2)}, \boldsymbol{b}^{(2)},\frac{∂l}{∂\boldsymbol{\hat{y}}},\frac{∂\boldsymbol{z}2}{∂\boldsymbol{z}1}, \frac{∂\boldsymbol{\hat{y}}}{∂\boldsymbol{z}_3}$ in your answer.**

| Layer                 | Input                | Output        |
| :---                  |    :----:            |          ---: |
| $\text{Linear}_1$     | $\boldsymbol{x}$     |  $\boldsymbol{z}_1 = \boldsymbol{W^{(1)}}\boldsymbol{x} + \boldsymbol{b^{(1)}}$  |
| $f$                   | $\boldsymbol{z}_1$        | $\boldsymbol{z}_2 = \text{ReLU}(\boldsymbol{z}_1)$     |
| $\text{Linear}_2$     | $\boldsymbol{z}_2$      | $\boldsymbol{z}_3 = \boldsymbol{W^{(2)}}\boldsymbol{z}_2 + \boldsymbol{b^{(2)}}$   |
| $g$                   | $\boldsymbol{z}_3$        |   $\boldsymbol{\hat{y}} = \boldsymbol{z}_3$    |
| Loss                  | $\boldsymbol{\hat{y}}, \boldsymbol{y}$       | $\Vert \boldsymbol{y} - \boldsymbol{\hat{y}} \Vert^2$ |