from mpi4py import MPI
from mat import gaussian_kernel_matrix
import pandas as pd
import numpy as np
from utils import auto_gc
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size()
ridge_lambda = 1.0 #np.power(np.float64(10), -1)
ridge_gamma = 1.0

# ridge_s = 

def log(*args):
    if rank == 0: print(args)

def loga(*args): print(f"Rank {rank}:", args)

def read_data(name, rank):
    file_name = f"data/{name}_{rank}.tsv"
    df = pd.read_csv(file_name, sep='\t', header=None)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y 

x_train, y_train = read_data("housing_train", rank)
x_test, y_test = read_data("housing_test", rank)

N = np.empty(1, dtype="int32")
comm.Allreduce(np.int32(x_train.shape[0]), N, op=MPI.SUM)
N = N[0]

N_test = np.empty(1, dtype="int32")
comm.Allreduce(np.int32(x_test.shape[0]), N_test, op=MPI.SUM)
N_test = N_test[0]

# log(type(N), N)

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

x_sum, y_sum = get_col_sum(x_train), get_col_sum(y_train)
x_avg, y_avg = get_avg(x_sum), get_avg(y_sum)
x_ss, y_ss = get_sigmasq(x_train, x_avg), get_sigmasq(y_train, y_avg)

# log(x_train[:3, :])
x_train, y_train = (x_train - x_avg) / np.sqrt(x_ss), (y_train - y_avg) / np.sqrt(y_ss)
x_test = (x_test - x_avg) / np.sqrt(x_ss)
# y_test = (y_test - y_avg) / np.sqrt(y_ss)
# log(X[:2, :], Y[:2])
# log(x_avg, x_ss, y_avg, y_ss)
# Above has been verified

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=17)
# loga(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

n_sample, n_feature = x_train.shape[0], x_train.shape[1]
r_x_train, r_y_train = [], []
for i in range(n_sample):
    row = []
    for j in range(n_feature):
        row.append(x_train[i][j])
    r_x_train.append(row)

for i in range(n_sample): r_y_train.append(y_train[i])

x_train, y_train = np.array(r_x_train), np.array(r_y_train)

r_x_test, r_y_test = [], []
n_sample_test = x_test.shape[0]
for i in range(n_sample_test): r_y_test.append(y_test[i])
for i in range(n_sample_test):
    row = []
    for j in range(n_feature):
        row.append(x_test[i][j])
    r_x_test.append(row)
x_test, y_test = np.array(r_x_test), np.array(r_y_test)

requests = [None] * n_proc
mat_block = [None] * n_proc

all_n_sample = np.empty(n_proc, dtype='int32')
comm.Allgather([np.int32(n_sample), MPI.INT], [all_n_sample, MPI.INT])
displs = np.copy(all_n_sample)
displs[0] = 0
for i in range(1, n_proc): displs[i] = displs[i-1] + all_n_sample[i-1]

# N_train = np.sum(all_n_sample)
# N_test = N - N_train

# log(type(N), N)
# log(type(N_train), N_train)
# log(type(N_test), N_test)

# loga(x_train.shape)
# loga(y_train.shape)

def barrier():
    confirm_signal = np.array(1, dtype='int32')
    all_signals = np.empty(n_proc, dtype='int32')
    comm.Allgather(confirm_signal, all_signals)

@auto_gc()
def distribute_mat_block(round, x_local, mat_block):
    if rank == round:
        data = x_train
        shape = x_train.shape
    else:
        data = np.empty((all_n_sample[round],n_feature))
        shape = None
    # if rank == 1 and round == 1:
    #     print(data.shape)
    #     print(data)
    # shape = comm.bcast(shape, root=round)
    # if rank != round: data = np.empty(shape, dtype="float64")
    # if (rank == 0 and round == 0):
    #     log(data.shape)
    #     log(data)
    comm.Bcast(data, root=round)

    # loga(round, data)
    # mat_block[round] = rbf_kernel(x_train, data, gamma=1/(2*ridge_s**2))
    mat_block[round] = rbf_kernel(x_local, data, gamma=ridge_gamma)

    # log(mat_block[round][:5,:5])
    # print(f"Round = {round}", mat_block[round][:10, :10])
    barrier()

for round in range(n_proc): distribute_mat_block(round, x_train, mat_block)
# loga(x_train)
k_mat = np.hstack(mat_block)

# log(x_train[:3, :])
# log(k_mat[:3, :3])
for i in range(k_mat.shape[0]): 
    k_mat[i][i+displs[rank]] = np.add(k_mat[i][i+displs[rank]], ridge_lambda)

threshold = np.power(np.float64(10), -6)
alpha = np.ones(n_sample)
# alpha = np.random.rand(n_sample)

# print(y.shape, alpha.shape)
@auto_gc()
def get_full(vec):
    full_vec = np.empty(N, dtype="float64")
    counts = np.array(all_n_sample, dtype="int32")
    displs_ = np.array(displs, dtype="int32")
    # print(counts, displs)
    comm.Allgatherv(sendbuf=vec, recvbuf=(full_vec, counts, displs_, MPI.DOUBLE))
    return full_vec

full_alpha = get_full(alpha)
# log(full_alpha)
# loga(k_mat)
# print(full_alpha.shape)
# print((k_mat @ full_alpha).shape)
r = np.subtract(y_train, k_mat @ full_alpha)
# log(full_alpha)
# log(k_mat @ full_alpha)
# loga(r)
p = np.copy(r)

def inner_product(a, b):
    sum_data = np.zeros(1, dtype="float64")
    comm.Allreduce([np.dot(a, b), MPI.DOUBLE], [sum_data, MPI.DOUBLE], op=MPI.SUM)
    return sum_data

se = inner_product(r, r)
# log(se)
# loga(p)
# if not np.less(se, threshold): log("Ha!")
it = 0

while True:
    if np.less(se, threshold): break  

    full_p = get_full(p)
    w = k_mat @ full_p
    # log(full_p)

    s = np.divide(se, inner_product(p, w)) 
    # log(s)
    # print(r.shape, w.shape, s.shape, p.shape, alpha.shape)
    # print(np.multiply(s, p).shape, np.multiply(s, w).shape)
    alpha = np.add(alpha, np.multiply(s, p))
    r = np.subtract(r, np.multiply(s, w))
    new_se = inner_product(r, r)
    beta = np.divide(new_se, se)
    # print(new_se.shape, se.shape, beta.shape)
    p = np.add(r, np.multiply(beta, p))
    # print(p.shape)
    se = new_se
    # log(se)

full_alpha = get_full(alpha)

# # if rank == 0:
# #     print(full_alpha.shape)
# #     print(full_alpha[:10])

# log(k_eval.shape)
# log(alpha.shape)
# log(pred_y_test.shape)

test_mat_block = [None] * n_proc
for round in range(n_proc): distribute_mat_block(round, x_test, test_mat_block)
k_mat_test = np.hstack(test_mat_block)
pred_y_test = k_mat_test @ full_alpha * np.sqrt(y_ss) + y_avg

mse_partial = np.sum(np.divide((y_test - pred_y_test) ** 2, N_test))
mse = np.empty(1) if rank == 0 else None
comm.Reduce(mse_partial, mse, op=MPI.SUM, root=0)
if rank == 0: 
    rmse = np.sqrt(mse)
    log(f"RMSE = {rmse[0]}")