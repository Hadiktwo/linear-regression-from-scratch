# Multiple Linear Regression from Scratch

A hands-on implementation of **Multiple Linear Regression** built from scratch with **NumPy**, without using Scikit-learn for the actual model training.

The goal of this project is to understand what happens inside a linear regression model instead of treating it as a black box. The implementation covers the forward pass, gradient calculation, gradient descent optimization, loss tracking, weight persistence, and evaluation with MSE and R².

## Project Overview

The project uses an advertising dataset with three input features:

- **TV** advertising budget
- **Radio** advertising budget
- **Newspaper** advertising budget

The target variable is:

- **Sales**

The dataset is divided into:

- **160 training samples**
- **40 test samples**

## What This Project Implements

The main parts of the model are implemented manually:

1. Multiple Linear Regression forward computation
2. Mean Squared Error (MSE)
3. Gradient calculation
4. Gradient Descent optimization
5. Bias/intercept handling
6. Training-loss tracking
7. Weight saving and loading with NumPy
8. Test-set evaluation using MSE and R²

## Model

The model follows the standard multiple linear regression equation:

$$
\hat{y} = w_0 + w_1x_1 + w_2x_2 + w_3x_3
$$

where:

- $w_0$ is the bias/intercept
- $w_1$ corresponds to TV
- $w_2$ corresponds to Radio
- $w_3$ corresponds to Newspaper

To include the bias in the matrix calculation, a column of ones is added to the feature matrix:

```python
x_train = np.hstack((np.ones((n, 1)), x_train))
```

## Training Process

The model is optimized using Gradient Descent.

At each epoch:

```text
1. Calculate predictions
2. Calculate MSE loss
3. Calculate gradients
4. Update weights
5. Store the loss
```

The update rule is:

$$
w := w - \eta \nabla J(w)
$$

where:

- $\eta$ is the learning rate
- $\nabla J(w)$ is the gradient of the loss function

### Training Configuration

| Parameter | Value |
|---|---:|
| Learning rate | 0.01 |
| Epochs | 1000 |
| Loss function | MSE |
| Optimizer | Gradient Descent |
| Features | 3 |

## Main Functions

### Linear Regression

```python
def linear_regression(x, w):
    y_hat = 0
    for wi, xi in zip(w, x.T):
        y_hat += xi * wi
    return y_hat
```

### Mean Squared Error

```python
def mse(y, y_hat):
    loss = np.mean((y - y_hat) ** 2)
    return loss
```

### Gradient

```python
def gradient(x, y, y_hat):
    grads = []
    for xi in x.T:
        grads.append(2 * np.mean(xi * (y_hat - y)))
    return np.array(grads)
```

### Gradient Descent

```python
def gradient_descent(w, eta, grads):
    w -= eta * grads
    return w
```

## Training Result

After 1000 epochs, the training loss converges to approximately:

```text
MSE ≈ 0.1088
```

The learned weights recorded in the notebook are approximately:

```text
bias        ≈ 0.000000001
TV weight   ≈ 0.76847
Radio       ≈ 0.50532
Newspaper   ≈ 0.00134
```

This result also provides an interesting observation: in this dataset, the learned coefficients for **TV** and **Radio** are much stronger than the coefficient for **Newspaper**.

## Evaluation

The notebook evaluates the model using:

### Mean Squared Error

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

### R² Score

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

The current notebook records:

```text
R²  ≈ 0.148
MSE ≈ 0.760
```

## Important Implementation Note

The current notebook contains a **test-time preprocessing issue** that affects the reported R² and MSE.

During training, a bias column is added:

```python
x_train = np.hstack((np.ones((n, 1)), x_train))
```

However, before test prediction the same bias column is not added. The test data therefore has 3 columns while the learned weight vector has 4 values.

The test pipeline should also include the bias term:

```python
x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
y_hat_test = linear_regression(x_test, w)
```

This is an important machine-learning lesson: **the preprocessing and feature representation used during training must be consistent with the representation used during inference.**

## Why Build It From Scratch?

Although libraries such as Scikit-learn can train Linear Regression models in a few lines, implementing the algorithm manually helps build a deeper understanding of:

- model parameters
- bias/intercept
- loss functions
- gradients
- optimization
- matrix operations
- training convergence
- model evaluation

This project is therefore primarily an **educational implementation**, not a production-ready regression package.

## Visualizations

The notebook includes:

- Scatter plots of **TV vs Sales**
- Scatter plots of **Radio vs Sales**
- Scatter plots of **Newspaper vs Sales**
- Training loss across epochs

These visualizations are used to understand relationships between the features and target, as well as to observe whether the optimization process is converging.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

Scikit-learn is imported in the notebook for reference, but the actual regression model in this project is implemented manually.

## Project Structure

```text
.
├── hal_tamirin_1.ipynb
├── train_advertising_data.csv
├── test_advertising_data.csv
└── multiple-linear-regression-weights.npy
```

> The CSV and `.npy` files are part of the expected project structure used by the notebook.

## How to Run

1. Clone the repository.
2. Place the training and test CSV files in the same directory as the notebook.
3. Open `hal_tamirin_1.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells from top to bottom.
5. Apply the test-time bias fix described above before evaluating the model.

## Learning Outcomes

By completing this project, I practiced implementing a machine-learning algorithm without relying on a high-level ML library and gained practical experience with:

- NumPy-based numerical computation
- custom model functions
- gradient descent
- debugging model evaluation
- tracking training loss
- saving model parameters
- regression metrics

## Next Improvements

Possible next steps for improving the project include:

- replacing the manual loops with vectorized NumPy operations
- adding a clean train/test preprocessing pipeline
- comparing the custom implementation with Scikit-learn
- experimenting with different learning rates and epoch counts
- adding MAE, RMSE, and adjusted R²
- plotting predictions against actual values
- adding assertions to prevent feature/weight dimension mismatches

## Author

**Hadi Karamzadeh**

Computer Engineering Student | Interested in Data Analysis, Machine Learning, and Practical AI
