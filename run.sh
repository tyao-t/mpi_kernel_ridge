#!/bin/bash

error_exit() {
    echo "$1" >&2
    echo "Usage: $0 <n_proc>" >&2
    echo "<n_proc> must be a number." >&2
    return 1
}

if [ -z "$1" ]; then
    error_exit "Error: You must provide the n_proc argument."
fi

if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    error_exit "Error: n_proc must be a number."
fi

n_proc=$1

source script/clean.sh &&
python script/split_train_test.py &&
source script/split_data.sh $1 &&
mpirun -n $1 python src/kernel_ridge_cv.py &&
echo ""