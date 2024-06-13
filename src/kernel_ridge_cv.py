from mpi4py import MPI
from matrix_ops import gaussian_kernel_matrix
import pandas as pd
import numpy as np
from data_utils import auto_gc, read_data, rebuild, add_housing_pressure, process_ocean
from sklearn.metrics.pairwise import rbf_kernel
from skopt import BayesSearchCV
from skopt.space import Real

def log(*args):
    if MPI.COMM_WORLD.Get_rank() == 0: print(args)

def loga(*args): print(f"Rank {MPI.COMM_WORLD.Get_rank()}:", args)

class KernelRidgeMPI:
    def __init__(self, ridge_lambda=None, gamma=None):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_proc = self.comm.Get_size()
        self.ridge_lambda = ridge_lambda # np.power(np.float64(10), -1)
        self.gamma = gamma # self.gamma = (2 * (s**2)
    
    def set_params(self, ridge_lambda=None, gamma=None): 
        self.ridge_lambda = ridge_lambda
        self.gamma = gamma
        return self
    
    def get_params(self, deep=True): 
        return {"ridge_lambda": self.ridge_lambda, "gamma": self.gamma}
    
    def barrier(self):
        self.comm.barrier()
        return

        # Previous implementation
        confirm_signal = np.array(1, dtype='int32')
        all_signals = np.empty(self.n_proc, dtype='int32')
        self.comm.Allgather(confirm_signal, all_signals)

    # Standardization based on x_train and y_train (not using x_test or y_test)
    def standardization(self):
        def get_col_sum(x):
            x_sum = np.sum(x, axis=0)
            global_x_sum = np.zeros_like(x_sum)
            self.comm.Allreduce(x_sum, global_x_sum, op=MPI.SUM)
            return global_x_sum

        def get_avg(x_sum): return np.divide(x_sum, self.N)

        def get_sigmasq(x, x_avg):
            def sq_div_n(x):
                return np.multiply(np.divide(x, np.subtract(self.N, 1)), x)
            vectorized_sq_div_n = np.vectorize(sq_div_n)

            # Var(X) = E(X^2) - E(X)^2
            # Wanted to use the above equation but the square of sum(X) causes a numerical overflow
            x_mean_sq_sum = np.sum(vectorized_sq_div_n(np.subtract(x, x_avg)), axis=0)
            # print(x_mean_sq_sum)
            global_x_mean_sq = np.zeros_like(x_mean_sq_sum)
            self.comm.Allreduce(x_mean_sq_sum, global_x_mean_sq, op=MPI.SUM)
            return global_x_mean_sq

        x_sum, y_sum = get_col_sum(self.x_train), get_col_sum(self.y_train)
        self.x_avg, self.y_avg = get_avg(x_sum), get_avg(y_sum)
        self.x_ss, self.y_ss = get_sigmasq(self.x_train, self.x_avg), get_sigmasq(self.y_train, self.y_avg)

        self.x_train, self.y_train = (self.x_train - self.x_avg) / np.sqrt(self.x_ss), (self.y_train - self.y_avg) / np.sqrt(self.y_ss)

    """ Function for implementing the round (robin) algorithm as in Lecture 4 slides 31-34 \
        each call of this function emulates a round. """
    @auto_gc()
    def distribute_mat_block(self, round, x_local, mat_block):
        data = self.x_train if self.rank == round else np.empty((self.all_n_sample[round],self.n_feature))
        self.comm.Bcast(data, root=round)
        # mat_block[round] = gaussian_kernel_matrix(self.x_train, data, self.gamma=1/(2*ridge_s**2))
        mat_block[round] = rbf_kernel(x_local, data, gamma=self.gamma)
        self.barrier()

    """ The reason I named this function "scale_income" was that: \
        at first I only wanted to scale income after standardization, \
        but I ended up scaling a number of variables back. However, \
        I decided not to change the function name because it sounds intuitive. """
    def scale_income(self, x): 
        for i in range(x.shape[0]): 
            x[i][0] *= 29 # Horizontal coordinate
            x[i][1] *= 31  # Vertical coordinate
            x[i][5] *= 0.125 # population
            x[i][6] *= 1.75 # population / households
            x[i][8] *= 1.75 # median income
            # x[i][3] *= 1.2 # Total rooms / population
            # x[i][4] *= 1.2 # Total bedrooms / population

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        N = np.empty(1, dtype="int32")
        self.comm.Allreduce(np.int32(self.x_train.shape[0]), N, op=MPI.SUM)
        self.N = N[0] # N is N_train
        # log(N) 

        # Processing before standardization
        ### Box-Cox (scale forward)
        # # # y_train_full_boxcox = get_full(y_train)
        # # # if rank == 0:
        # # #     pt = PowerTransformer(method='box-cox', standardize=True)
        # # #     y_train_full_boxcox = pt.fit_transform(y_train_full_boxcox.reshape(-1, 1)).ravel()
        # # # comm.Bcast(y_train_full_boxcox, root=0)
        # # # y_train = rebuild(y_train_full_boxcox[displs[rank]:displs[rank]+n_sample].ravel())
        process_ocean(self.x_train, self.y_train, self.comm)
        add_housing_pressure(self.x_train)

        self.standardization()   
        # Processing after standardization
        self.scale_income(self.x_train)

        self.x_train, self.y_train = rebuild(self.x_train), rebuild(self.y_train) 
        n_sample, self.n_feature = self.x_train.shape[0], self.x_train.shape[1]

        # Start computing the Kernel matrix using the round (robin) method
        mat_block = [None] * self.n_proc # K's mat blocks

        # Preparation, for (All)Gatherv
        self.all_n_sample = np.zeros(self.n_proc, dtype='int32')
        self.comm.Allgather([np.int32(n_sample), MPI.INT], [self.all_n_sample, MPI.INT])
        displs_ = np.zeros(self.n_proc, dtype='int32') # Displacements
        for i in range(1, self.n_proc): displs_[i] = displs_[i-1] + self.all_n_sample[i-1]
        counts = np.array(self.all_n_sample, dtype="int32")

        @auto_gc()
        def get_full(vec):
            full_vec=np.empty(self.N, dtype="float64")
            self.comm.Allgatherv(sendbuf=vec, recvbuf=(full_vec, counts, displs_, MPI.DOUBLE))
            return full_vec

        # Start of round (robin)
        for round in range(self.n_proc): self.distribute_mat_block(round, self.x_train, mat_block)
        k_mat = np.hstack(mat_block)
        # End of round (robin)

        # Add the digonal entry
        for i in range(k_mat.shape[0]): 
            k_mat[i][i+displs_[self.rank]] = np.add(k_mat[i][i+displs_[self.rank]], self.ridge_lambda)

        def conjugate_gradient():
            threshold = np.power(np.float64(10), -6)
            alpha = np.random.rand(n_sample)

            self.full_alpha = get_full(alpha)
            r = np.subtract(self.y_train, k_mat @ self.full_alpha)
            p = np.copy(r)

            def inner_product(a, b):
                sum_data = np.zeros(1, dtype="float64")
                self.comm.Allreduce([np.dot(a, b), MPI.DOUBLE], [sum_data, MPI.DOUBLE], op=MPI.SUM)
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

        self.full_alpha = conjugate_gradient()

    def score(self, x_test, y_test):
        # Preprocess and transformations
        process_ocean(x_test, None, self.comm)
        add_housing_pressure(x_test)
        x_test, y_test = (x_test - self.x_avg) / np.sqrt(self.x_ss), y_test #(y_test - self.y_avg) / np.sqrt(self.y_ss)
        x_test, y_test = rebuild(x_test), rebuild(y_test)
        self.N_test = np.empty(1, dtype="int32")
        self.comm.Allreduce(np.int32(x_test.shape[0]), self.N_test, op=MPI.SUM)
        self.N_test = self.N_test[0] # n_sample_test = x_test.shape[0]
        self.scale_income(x_test) 

        # Compute the prediction for the testing x's
        test_mat_block = [None] * self.n_proc
        for round in range(self.n_proc): self.distribute_mat_block(round, x_test, test_mat_block)
        k_mat_test = np.hstack(test_mat_block)
        pred_y_test = k_mat_test @ self.full_alpha * np.sqrt(self.y_ss) + self.y_avg

        ### Box-Cox (scale back)
        # # # pred_y_test = k_mat_test @ self.full_alpha
        # # # pred_y_test_full = get_full(pred_y_test)
        # # # if self.rank == 0: pred_y_test_full = pt.inverse_transform(pred_y_test_full.reshape(-1, 1)).ravel()
        # # # self.comm.Bcast(pred_y_test_full, root=0)
        # # # y_test_full = get_full(y_test)

        # Compute RMSE
        mse_partial = np.sum(np.divide((y_test - pred_y_test) ** 2, self.N_test))
        mse = np.empty(1)
        self.comm.Allreduce(mse_partial, mse, op=MPI.SUM)
        rmse = np.sqrt(mse)

        return -rmse


def main():
    kmpi = KernelRidgeMPI(0.1037, 0.0912)
    x_train, y_train = read_data("housing_train", kmpi.rank)
    x_test, y_test = read_data("housing_test", kmpi.rank)

    def single():   
        kmpi.fit(x_train, y_train)
        rmse = -kmpi.score(x_test, y_test)
        log(rmse)
    
    single()

    def cv():
        search_space = {
            'ridge_lambda': Real(0.008, 0.32, prior='log-uniform'),
            'gamma': Real(0.008, 0.32, prior='log-uniform')
        }

        bayes_search = BayesSearchCV(
            estimator=KernelRidgeMPI(),
            search_spaces=search_space,
            n_iter=64,
            random_state=45,
            cv=5,
            n_jobs=1,
            verbose=1
        )

        bayes_search.fit(x_train, y_train)

        log(bayes_search.best_score_)
        log(bayes_search.best_params_)
        log(bayes_search.best_estimator_.score(x_test, y_test))

    # cv()

if __name__=="__main__":
    main()
