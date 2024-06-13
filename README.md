# Kernel Ridge Regression with MPI

## TL;DR

- This project implements Kernel Ridge Regression using MPI with some feature engineering (important) and hyper-parameter tuning. The algorithm of choice is the conjugate gradient method, as taught in Lecture 4, along with the round-robin method discussed from page 31 to page 34 of Lecture 4's slides. Memory optimization is applied wherever possible, both in these methods and other parts of the implementation; Worked as an individual. 

- Achieved 43948.3 RMSE on the test set (random_state=17) and a RMSE between ~44000 + 0 ~ +6000 in general (for other train-test splits).

## Requirements

- mpich or another MPI library
- Python 3.9+
- Dependencies as specified in requirements.txt

## Installation

If MPICH is not installed, please execute the following commands:

```sh
sudo apt-get update
sudo apt-get install mpich
```

## Execution

```sh 
source ./run.sh <num_processes>
```

## Additional Notes

Code compatible with (can be supplied as a custom model into) scikit-learn's GridSearchCV and BayesSearchCV, given that `__init__`, `fit`, and `score` interface functions are implemented.