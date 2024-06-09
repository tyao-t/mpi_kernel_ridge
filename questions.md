
# Tentative Steps

## Data Points

```bash
(mpi) Tianhaos-MacBook:mpi_kernel_ridge tianhaoyao$ wc -l data/cal_housing.data 
   20640 data/cal_housing.data
(mpi) Tianhaos-MacBook:mpi_kernel_ridge tianhaoyao$ wc -l data/housing.tsv 
   20433 data/housing.tsv
```

**Is this because ocean proximity was unavailable for 20640-20433 = 207 data points, so these 207 rows have been removed? Just to confirm that we should just use the tsv dataset which contains 20433 rows rather than the other.**

## Feature Engineering (You can skip some or all for now; Data Points and Standardization are the two most important sections for questions)

- One hot encoding for ocean proximity (not doing for now, since it doesn't make too much of a difference)
- **Need log(Y) or Box-cox transformation for Y or some predictors? Not too familiar with the dataset so I am not completely sure at this point. Haven't done anything yet.**
- **Need to consider for interaction effect or anything like that? I guess not for now...**
- **PCA or other ways of reducing dimensionality? I guess not for now either...**

## Standardization

My approach is to calculate the mean and variance of the 70% training set rather than everything, and use this mean and variance to also standardize the 30% testing set. So basically the testing set's mean and variance was never calculated or considered. This method is recommended by various online sources including AI and non-AI. Do you think it is correct and/or canonical? 

## Results

Put in a huge amount of effort to write and debug the programs, eliminating a bunch of human and around 2 esoteric library errors.

```bash
(mpi) Tianhaos-MacBook:mpi_kernel_ridge tianhaoyao$ ./run.sh 1 && ./run.sh 2 && ./run.sh 4 && ./run.sh 8
Running with 1 process
('RMSE = 57257.16941379172',)

Running with 2 processes
('RMSE = 57257.17794661127',)

Running with 4 processes
('RMSE = 57257.17753791768',)

Running with 8 processes
('RMSE = 57257.176756067136',)

(mpi) Tianhaos-MacBook:mpi_kernel_ridge tianhaoyao$ !py
python sklearn_rmse.py 
RMSE: 57257.59190363003
```

## Code Readability

Will improve very soon.
