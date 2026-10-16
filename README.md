# Multiple Linear Regression from Scratch

This repository contains a custom Python implementation of a Multiple Linear Regression (MLR) model built entirely from scratch using matrix algebra. The mathematical model is benchmarked against Scikit-Learn's `LinearRegression` class to validate its accuracy and exactness.

## Overview

Instead of using gradient descent, this implementation utilizes the **Ordinary Least Squares (OLS) Normal Equation** to calculate the optimal parameters in a single step. 

By inserting a bias column of 1s into the feature matrix, the model computes the combined coefficients and intercept using linear algebra:
`Beta = (X^T * X)^-1 * X^T * y`

The custom class heavily leverages `numpy.linalg.inv` and vector dot products to solve the equation efficiently.

## Dataset

The models are trained and tested on the built-in Scikit-Learn **Diabetes Dataset**.
* **Features (X):** 10 physiological baseline variables.
* **Target (y):** A quantitative measure of disease progression one year after baseline.

The dataset is partitioned into a training set and a testing set utilizing a random state for strict reproducibility.

## Results & Validation

The custom NumPy matrix implementation flawlessly matches Scikit-Learn's model, yielding identical intercepts, feature coefficients, and predictive R-squared scores. You can run the included Python script or Jupyter Notebook to see the side-by-side terminal output verifying that both models compute the exact same mathematical parameters.

## Requirements

To run the script locally, ensure you have the following dependencies installed:
* `numpy`
* `pandas`
* `scikit-learn`

## Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
