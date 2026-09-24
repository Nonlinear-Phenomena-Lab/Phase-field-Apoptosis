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
        nx, nz = Nx, Nz  # Assuming gathered shape for XDMF generation
        lx, lz = Lx, Lz
    else:
        nx, nz = int(1.5 * Nx), int(1.5 * Nz)
        lx = 1.0 * Lx
        lz = 1.0 * Lz

    return nx, nz, lx, lz

def save_distributed_array(Nx, Nz, domain, dist, local_array, file_name, full_shape, binary_format='bin'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Recalculate the full shape
    full_shape = find_global_shape(local_array.T)
    #print(f"Global shape: {full_shape}")
    
    # Recalculate slices
    scales = 1
    slices = dist.grid_layout.slices(domain, scales=scales)
    #print(f"Slices on rank {rank}: {slices}")

    # Gather the local arrays and slices
    local_data = (local_array.copy(), slices)
    gathered_data = comm.gather(local_data, root=0)

    if rank == 0:
        full_shape_T = full_shape
        full_shape = (full_shape_T[1], full_shape_T[0])
        full_array = np.zeros(full_shape, dtype=local_array.dtype)

        for data, slc in gathered_data:
            if full_shape[0] == Nx or full_shape[1] == Nz:
                full_array[slc] = data  # Transpose if necessary
            else:
                #print(f"Scaling slices: {slc}")
                scaled_slices = tuple(
                    slice(int(1.5 * s.start), int(1.5 * s.stop), s.step) for s in slc
                )
                full_array[scaled_slices] = data  # Transpose if necessary

        # Save the full array to file
        full_array = full_array
        full_array.tofile(file_name + '.' + binary_format)
        #print(f"Array saved successfully to {file_name}.{binary_format}")

    comm.Barrier()

def save_distributed_vector(Nx, Ny, domain, dist, local_vector, file_name, full_shape, binary_format='bin'):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Recalculate the full shape
    full_shape = find_global_shape(local_vector.T)
    #print(f"Global shape: {full_shape}")

    # Recalculate slices
    scales = 1
    slices = dist.grid_layout.slices(domain, scales=scales)
    #print(f"Slices on rank {rank}: {slices}")

    # Gather the local vectors and slices
    local_data = (local_vector.copy(), slices)
    gathered_data = comm.gather(local_data, root=0)

    #print(rank, slices)

    if rank == 0:
        # Adjust the full shape to include the vector dimension
        full_shape_T = full_shape
        #full_shape = (full_shape_T[1], full_shape_T[0], 2)  # Assuming the vector has 2 components
        full_vector = np.zeros(full_shape, dtype=local_vector.dtype)

        for data, slc in gathered_data:
            #print(full_shape)
            #print(full_vector.shape)
            #print(data.shape)
            if full_shape[0] == Nx or full_shape[1] == Ny:
                full_vector[slc] = data  # Transpose if necessary
            else:
                #print(f"Scaling slices: {slc}")
                scaled_slices = tuple(
                    slice(int(1.5 * s.start), int(1.5 * s.stop), s.step) for s in slc
                )
                full_vector[scaled_slices] = np.transpose(data, (1, 2, 0))  # Transpose if necessary
                #full_vector = np.flip(full_vector, axis=0)

        # Save the full vector to file
        full_vector = full_vector
        full_vector.tofile(file_name + '.' + binary_format)

    comm.Barrier()

def find_global_shape(local_array):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get the shape of the local array on each process
    local_shape = np.array(local_array.shape)

    # Gather all shapes on rank 0
    all_shapes = None
    if rank == 0:
        all_shapes = np.empty((size, len(local_shape)), dtype=int)  # To store shapes of all processes

    comm.Gather(local_shape, all_shapes, root=0)

    # Rank 0 calculates the global shape
    if rank == 0:
        # Calculate global shape:
        # Assume the first dimension varies (split among processes), sum it
        # All other dimensions should be the same, so take them from any local shape (e.g., the first one)
        #print(all_shapes)
        global_shape = list(all_shapes[0])
        global_shape[0] = np.sum(all_shapes[:, 0])  # Sum the first dimension (assuming it's split)
        global_shape = tuple(global_shape)
        return global_shape
