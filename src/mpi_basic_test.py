from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Root process creates a 3x2 array
if rank == 0:
    array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(array.shape)
else:
    array = None

# Determine array shape and dtype, root process broadcasts this information to other processes
if rank == 0:
    shape = array.shape
    dtype = array.dtype
else:
    shape = None
    dtype = None

shape = comm.bcast(shape, root=0)
dtype = comm.bcast(dtype, root=0)

# All processes create an empty array of the same shape and dtype
if rank != 0:
    # print(shape)
    array = np.empty(shape, dtype=dtype)

# Broadcast array data from root process to all other processes
comm.Bcast(array, root=0)

# Each process prints the received array
print(f"Process {rank} received array:\n{array}")

# Please note that this code is designed to be run in an MPI environment.
# You should save this code to a file, for example, `broadcast_array.py`, and run it using an MPI command:
# mpirun -n 4 python broadcast_array.py