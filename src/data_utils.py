import gc
from functools import wraps
import numpy as np
import pandas as pd
from mpi4py import MPI

def auto_gc(enabled=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if enabled:
                gc.collect()
            return result
        return wrapper
    return decorator

@auto_gc()
def read_data(name, rank):
    file_name = f"data/{name}_{rank}.tsv"
    df = pd.read_csv(file_name, sep='\t', header=None)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y 

def add_housing_pressure(x):
    for i in range(x.shape[0]):
        x[i][3] /= x[i][5]
        x[i][4] /= x[i][5]
        x[i][6] = x[i][5]/x[i][6]

def rebuild(x):
    r_x = np.ndarray(x.shape)
    for index, element in np.ndenumerate(x):
        r_x[index] = element

    return r_x
    # # # if x.ndim == 1:
    # # #     r_x = np.array(x.shape[0]) # Can think of it as real_x or rebuilt_x
    # # #     for i in range(x.shape[0]): r_x[i] = x[i]
    # # #     return r_x
    # # # elif x.ndim == 2:
    # # #     r_x = [] 
    # # #     for i in range(x.shape[0]):
    # # #         row = []
    # # #         for j in range(x.shape[1]):
    # # #             row.append(x[i][j])
    # # #         r_x.append(row)
    # # #     return np.array(r_x)

ocean_avg = None
def process_ocean(x, y, comm):
    default = {0: 1, 1: -1, 2: 1.1, 3: 1.2, 4: 1.5}
    def get_ocean_y_avg(x, y):
        ocean_y_sum = [0] * 5
        ocean_count = [0] * 5
        for i in range(x.shape[0]): 
            ocean_y_sum[int(x[i][8]+0.9)] += y[i]
            ocean_count[int(x[i][8]+0.9)] += 1
        ocean_y_sum = np.array(ocean_y_sum)
        ocean_count = np.array(ocean_count)
        global_ocean_y_sum = np.zeros_like(ocean_y_sum)
        global_ocean_count = np.zeros_like(ocean_count)
        comm.Allreduce(ocean_y_sum, global_ocean_y_sum, op=MPI.SUM)
        comm.Allreduce(ocean_count, global_ocean_count, op=MPI.SUM)
        if i in range(5): 
            if global_ocean_count[i] == 0: return default
        return global_ocean_y_sum / global_ocean_count

    global ocean_avg
    # ocean_avg = default
    if ocean_avg is None: ocean_avg = get_ocean_y_avg(x, y).ravel()
    for i in range(x.shape[0]): x[i][8] = ocean_avg[int(x[i][8]+0.9)]