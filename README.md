
# Kernel Ridge Regression with MPI

## Overview

- This project implements Kernel Ridge Regression using MPI with some feature engineering (important) and hyper-parameter tuning. The algorithm of choice is the conjugate gradient method, along with a round-robin method. Optimization regarding memory consumption is applied wherever possible, both in these methods and in other parts of the implementation; Worked as an individual.

- Achieved 43,948.3 RMSE on the test set (random_state=17) and a RMSE between ~44,000 + ~5,000, guaranteed for other train-test splits.

## Structure

The project involves the following steps:
1. Data splits cleaning and re-splitting using a script.
2. MPI-based preprocessing and gathering/broadcasting/reduction of data summaries for each process.
3. Calculation of the Kernel Matrix.
4. Application of the conjugate gradient method.
5. Prediction for x_test.

Each step includes reduction operations, typically involving the sum, followed by division to average. Allgather operations abstract some of these processes. For more details, please refer to the code in `kernel_ridge_cv.py` in the `src` directory.

## Feature Engineering

Initially, after implementing kernel ridge regression, the RMSE was around 52,000. The goal was to reduce the RMSE to < 50,000, but this turned out to be impossible with sheer hyper-parameter tuning without changing the predictors. Consequently, several feature engineering techniques were applied. This process was incredibly educational and meaningful.

One critical adjustment involved the `OCEAN_PROXIMITY` feature. Observing the data revealed that houses with the lowest prices typically had a value of 1, while the highest prices had a value of 0. Categories 2, 3, and 4 were less common but generally had higher average prices than 0. To address this, the original values were replaced with the average Y value for each category using MPI. For example:

```python
{0: 240291.24, 1: 125374.74, 2: 247176.55, 3: 257555.42, 4: 293750.00}
```

Thus, the categories were reordered; values could also be manually adjusted as follows:

```python
ocean_avg = {0: 1, 1: -1, 2: 1.1, 3: 1.2, 4: 1.5}
```

This process is implemented in `process_ocean()`.

For predictors, `add_housing_pressure()` performs several steps. It normalizes `total_rooms` and `total_bedrooms` by the population to reflect housing pressure. The ratio of `population` to `households` was also created as a feature, replacing `households`, indicating areas where larger families might be present. Clustering of location coordinates (e.g. with K-means) was not used, as kernel regression inherently handles such relationships already. However, longitude and latitude weights were adjusted after standardization.

Additionally, feature importance was assessed using a random forest model (learned from another class), and predictors were scaled up or down accordingly, after standardization. For instance, `population` was identified as the least important feature. 

Conclusively, in `scale_income()`, weights for coordinates, housing pressure, and ocean proximity, etc. were scaled up based on these insights. These decisions were primarily inspired by personal experience with a small amount of guidance from AI tools.

### Omitted or Unused Feature Engineering

Box-Cox transformation was tested with sklearn and it offered similar benefits to simple standardization of Y. Due to numerical instability and increased complexity for code review, Box-Cox was ultimately commented out from the MPI implementation; but generally when Y is large, there is likely non-constant variance and non-uniformity. When Y is large (for example, when athletes have higher salaries, as in one example from another class), the variance increases. In these cases, transformations on Y, such as Box-Cox or square root transformations, are necessary. This is essentially the idea.

The implementation standardized the response variable Y using MPI, in order to enhance numerical and convergence stability, taking into account the fact that sklearn's Box-Cox transformation performs standardization on Y by default. Without standardization, the RMSE was approximately 48500 to 49000.

VIF was not used to check for multicollinearity, nor were any variables removed based on it.
Cook's distance or other outlier detection methods were not applied, as they were beyond the project's scope.

## Requirements

- MPICH or another MPI library installed
- Python 3.9+
- Dependencies as specified in `requirements.txt`

## Installation

If MPICH is not installed, execute the following commands:

```sh
sudo apt-get update
sudo apt-get install mpich
```

For MPI execution on multiple machines, additional setup is required, including data pre-distribution (e.g. using `scp`) and installing SSH server daemons (e.g. with systemd). This project as of now focuses on multi-process execution on a single machine.

## Execution

```sh 
source ./execute.sh <num_processes>
```

## Additional Notes

The code is fully compatible with scikit-learn's `GridSearchCV` and `BayesSearchCV` as the `__init__`, `fit`, and `score` interface functions have been implemented.

Random state 17 was first chosen as it was a student ID in primary school and a best friend's ID in middle school. Other random numbers, such as 9999, 46, or 397251, were arbitrarily selected.

MPI code execution in Jupyter notebooks was also attempted. This is theoretically possible with ipyparallel but not too viable in general because the programming (especially logging) paradigms radically change.
