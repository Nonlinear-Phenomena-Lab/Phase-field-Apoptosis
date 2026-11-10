import numpy as np
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def randomise(p):
    p_data = np.array(p['g'].copy())
    n = p_data.shape
    for i in range(n[0]):
        for j in range(n[1]):
            random_term = np.random.rand()
            if p_data[i][j] > 0.0 and p_data[i][j] < 0.5:
                p_data[i][j] = p_data[i][j] + 0.05*random_term
            else:
                p_data[i][j] = p_data[i][j]
    p['g'] = p_data.copy()     
    return p

def change_size(local_array, Nx, Nz, Lx, Lz):
    col_num = local_array.shape[0]
    print(col_num)
    if col_num == Nz:
        nx, nz = Nx, Nz
        lx, lz = Lx, Lz
    else:
        nx, nz = int(1.5 * Nx), int(1.5 * Nz)
        lx = 1.0 * Lx
        lz = 1.0 * Lz

    return nx, nz, lx, lz

def save_distributed_array(Nx, Nz, domain, dist, local_array, file_name, full_shape, binary_format='bin'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    full_shape = find_global_shape(local_array.T)
    
    scales = 1
    slices = dist.grid_layout.slices(domain, scales=scales)

    local_data = (local_array.copy(), slices)
    gathered_data = comm.gather(local_data, root=0)

    if rank == 0:
        full_shape_T = full_shape
        full_shape = (full_shape_T[1], full_shape_T[0])
        full_array = np.zeros(full_shape, dtype=local_array.dtype)

        for data, slc in gathered_data:
            if full_shape[0] == Nx or full_shape[1] == Nz:
                full_array[slc] = data
            else:
                scaled_slices = tuple(
                    slice(int(1.5 * s.start), int(1.5 * s.stop), s.step) for s in slc
                )
                full_array[scaled_slices] = data

        full_array = full_array
        full_array.tofile(file_name + '.' + binary_format)

    comm.Barrier()

def save_distributed_vector(Nx, Ny, domain, dist, local_vector, file_name, full_shape, binary_format='bin'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    full_shape = find_global_shape(local_vector.T)

    scales = 1
    slices = dist.grid_layout.slices(domain, scales=scales)

    local_data = (local_vector.copy(), slices)
    gathered_data = comm.gather(local_data, root=0)


    if rank == 0:
        full_shape_T = full_shape
        full_vector = np.zeros(full_shape, dtype=local_vector.dtype)

        for data, slc in gathered_data:
            if full_shape[0] == Nx or full_shape[1] == Ny:
                full_vector[slc] = data
            else:
                scaled_slices = tuple(
                    slice(int(1.5 * s.start), int(1.5 * s.stop), s.step) for s in slc
                )
                full_vector[scaled_slices] = np.transpose(data, (1, 2, 0))

        full_vector = full_vector
        full_vector.tofile(file_name + '.' + binary_format)

    comm.Barrier()

def find_global_shape(local_array):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_shape = np.array(local_array.shape)

    all_shapes = None
    if rank == 0:
        all_shapes = np.empty((size, len(local_shape)), dtype=int)

    comm.Gather(local_shape, all_shapes, root=0)

    if rank == 0:
        global_shape = list(all_shapes[0])
        global_shape[0] = np.sum(all_shapes[:, 0])
        global_shape = tuple(global_shape)
        return global_shape
