from mpi4py import MPI
from matrix_ops import gaussian_kernel_matrix
import pandas as pd
import numpy as np
from data_utils import auto_gc, read_data, rebuild, add_housing_pressure, process_ocean
from sklearn.metrics.pairwise import rbf_kernel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size()
ridge_lambda = 0.1037 # np.power(np.float64(10), -1)
gamma = 0.0912 # gamma = (2 * (s**2)

def log(*args):
    if rank == 0: print(args)

def loga(*args): print(f"Rank {rank}:", args)

x_train, y_train = read_data("housing_train", rank)
x_test, y_test = read_data("housing_test", rank)

N = np.empty(1, dtype="int32")
comm.Allreduce(np.int32(x_train.shape[0]), N, op=MPI.SUM)
N = N[0]
N_test = np.empty(1, dtype="int32")
comm.Allreduce(np.int32(x_test.shape[0]), N_test, op=MPI.SUM)
N_test = N_test[0]
# log(N, N_test)

# Processing before standardization
for x in x_train, x_test:
    process_ocean(x, y_train, comm)
    add_housing_pressure(x)

def standardization():
    def get_col_sum(x):
        x_sum = np.sum(x, axis=0)
        global_x_sum = np.zeros_like(x_sum)
        comm.Allreduce(x_sum, global_x_sum, op=MPI.SUM)
        return global_x_sum

    def get_avg(x_sum): return np.divide(x_sum, N)

    def get_sigmasq(x, x_avg):
        def sq_div_n(x):
            return np.multiply(np.divide(x, np.subtract(N, 1)), x)
        vectorized_sq_div_n = np.vectorize(sq_div_n)

        # Var(X) = E(X^2) - E(X)^2
        # Wanted to use the above equation but the square of sum(X) causes a numerical overflow
        x_mean_sq_sum = np.sum(vectorized_sq_div_n(np.subtract(x, x_avg)), axis=0)
        # print(x_mean_sq_sum)
        global_x_mean_sq = np.zeros_like(x_mean_sq_sum)
        comm.Allreduce(x_mean_sq_sum, global_x_mean_sq, op=MPI.SUM)
        return global_x_mean_sq

    global x_train, y_train, x_test, y_test, x_avg, x_ss, y_avg, y_ss
    x_sum, y_sum = get_col_sum(x_train), get_col_sum(y_train)
    x_avg, y_avg = get_avg(x_sum), get_avg(y_sum)
    x_ss, y_ss = get_sigmasq(x_train, x_avg), get_sigmasq(y_train, y_avg)

    x_train, y_train = (x_train - x_avg) / np.sqrt(x_ss), (y_train - y_avg) / np.sqrt(y_ss)
    x_test, y_test = (x_test - x_avg) / np.sqrt(x_ss), y_test #(y_test - y_avg) / np.sqrt(y_ss)
    # log(x_avg, x_ss, y_avg, y_ss)

standardization()

x_train, y_train = rebuild(x_train), rebuild(y_train) 
n_sample, n_feature = x_train.shape[0], x_train.shape[1]
x_test, y_test = rebuild(x_test), rebuild(y_test)
n_sample_test = x_test.shape[0]

mat_block = [None] * n_proc # K's mat blocks

all_n_sample = np.zeros(n_proc, dtype='int32')
comm.Allgather([np.int32(n_sample), MPI.INT], [all_n_sample, MPI.INT])
displs = np.zeros(n_proc, dtype='int32')
for i in range(1, n_proc): displs[i] = displs[i-1] + all_n_sample[i-1]
counts = np.array(all_n_sample, dtype="int32")
displs_ = np.array(displs, dtype="int32")

@auto_gc()
def get_full(vec):
    full_vec=np.empty(N, dtype="float64")
    comm.Allgatherv(sendbuf=vec, recvbuf=(full_vec, counts, displs_, MPI.DOUBLE))
    return full_vec

### Box-Cox (scale forward)
# # # y_train_full_boxcox = get_full(y_train)
# # # if rank == 0:
# # #     pt = PowerTransformer(method='box-cox', standardize=True)
# # #     y_train_full_boxcox = pt.fit_transform(y_train_full_boxcox.reshape(-1, 1)).ravel()
# # # comm.Bcast(y_train_full_boxcox, root=0)
# # # y_train = rebuild(y_train_full_boxcox[displs[rank]:displs[rank]+n_sample].ravel())

def scale_income(x): 
    for i in range(x.shape[0]): 
        x[i][0] *= 29 # Horizontal coordinate
        x[i][1] *= 31  # Vertical coordinate
        x[i][5] *= 0.125 # population
        x[i][6] *= 1.75 # population / households
        x[i][8] *= 1.75 # median income
        # x[i][3] *= 1.2 # Total rooms / population
        # x[i][4] *= 1.2 # Total bedrooms / population

""" The reason I named this function "scale_income" was that: \
    at first I only wanted to scale income after standardization, \
    but I ended up scaling a number of variables back. However, \
    I decided not to change the function name because it sounds intuitive. """
for x in x_train, x_test: scale_income(x)

def barrier():
    comm.barrier()
    return
    confirm_signal = np.array(1, dtype='int32')
    all_signals = np.empty(n_proc, dtype='int32')
    comm.Allgather(confirm_signal, all_signals)

@auto_gc()
def distribute_mat_block(round, x_local, mat_block):
    data = x_train if rank == round else np.empty((all_n_sample[round],n_feature))
    comm.Bcast(data, root=round)
    # mat_block[round] = gaussian_kernel_matrix(x_train, data, gamma=1/(2*ridge_s**2))
    mat_block[round] = rbf_kernel(x_local, data, gamma=gamma)
    barrier()

for round in range(n_proc): distribute_mat_block(round, x_train, mat_block)
k_mat = np.hstack(mat_block)

# Add the digonal entry
for i in range(k_mat.shape[0]): 
    k_mat[i][i+displs[rank]] = np.add(k_mat[i][i+displs[rank]], ridge_lambda)

def conjugate_gradient():
    threshold = np.power(np.float64(10), -6)
    alpha = np.random.rand(n_sample)

    full_alpha = get_full(alpha)
    r = np.subtract(y_train, k_mat @ full_alpha)
    p = np.copy(r)

    def inner_product(a, b):
        sum_data = np.zeros(1, dtype="float64")
        comm.Allreduce([np.dot(a, b), MPI.DOUBLE], [sum_data, MPI.DOUBLE], op=MPI.SUM)
        return sum_data

    se = inner_product(r, r)

    while True:
        if np.less(se, threshold): break  

        full_p = get_full(p)
        w = k_mat @ full_p

        s = np.divide(se, inner_product(p, w)) 
        alpha = np.add(alpha, np.multiply(s, p))
        r = np.subtract(r, np.multiply(s, w))
        new_se = inner_product(r, r)
        beta = np.divide(new_se, se)
        p = np.add(r, np.multiply(beta, p))
        se = new_se

    full_alpha = get_full(alpha)
    return full_alpha

full_alpha = conjugate_gradient()
# End of conjugate gradient method

# Once we got full alpha, compute the predictions then scale back
test_mat_block = [None] * n_proc
for round in range(n_proc): distribute_mat_block(round, x_test, test_mat_block)
k_mat_test = np.hstack(test_mat_block)
pred_y_test = k_mat_test @ full_alpha * np.sqrt(y_ss) + y_avg

### Box-Cox (scale back)
# # # pred_y_test = k_mat_test @ full_alpha
# # # pred_y_test_full = get_full(pred_y_test)
# # # if rank == 0: pred_y_test_full = pt.inverse_transform(pred_y_test_full.reshape(-1, 1)).ravel()
# # # comm.Bcast(pred_y_test_full, root=0)
# # # y_test_full = get_full(y_test)

# Compute RMSE
mse_partial = np.sum(np.divide((y_test - pred_y_test) ** 2, N_test))
mse = np.empty(1) if rank == 0 else None
comm.Reduce(mse_partial, mse, op=MPI.SUM, root=0)

if rank == 0: 
    rmse = np.sqrt(mse)
    log(f"RMSE = {rmse}")