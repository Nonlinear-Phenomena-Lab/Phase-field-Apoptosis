import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.core.system import *
from mpi4py import MPI
import logging
import utils as ut 
from initialisation import *

# Initialize MPI and logging
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_problem(Lx, Ly, Nx, Ny, l_f, l_phi, lr_phi, l1_sig, l2_sig, lr_sig, k2, dealias, dtype):
    import numpy as np

    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)
    
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx, Lx), dealias=dealias)
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly, Ly), dealias=dealias)

    x = dist.local_grids(xbasis)
    y = dist.local_grids(ybasis)

    phi = dist.Field(name='phi', bases=(xbasis,ybasis))
    ep = dist.Field(name='ep', bases=(xbasis,ybasis))
    ep_prime = dist.Field(name='ep_prime', bases=(xbasis,ybasis))
    sig = dist.Field(name='sig', bases=(xbasis,ybasis))
    r = dist.Field(name='r', bases=(xbasis,ybasis))
    theta = dist.Field(name='theta', bases=(xbasis,ybasis))
        
    dx = lambda A: d3.Differentiate(A, coords['x'])
    dy = lambda A: d3.Differentiate(A, coords['y'])

    vartheta = 25.0
    
    problem = d3.IVP([phi, ep, ep_prime, sig, r, theta], namespace=locals())
    problem.add_equation("(dt(phi)) + (l_f * (phi/2)) =  (l_phi * (-dx(ep * ep_prime * dy(phi)) + dy(ep * ep_prime * dx(phi)) + div((ep**2)*grad(phi)))) + (l_f * phi**2) - (l_f * phi**3) + (l_f * (phi**2)/2) + ((phi - phi**2) * lr_phi * r) ")
    problem.add_equation("dt(sig) - (l1_sig * lap(sig)) - (l2_sig * dt(phi)) = -lr_sig * r * (phi - (phi**2))")
    problem.add_equation("r = - np.arctan((k2 * sig))")
    problem.add_equation("theta = np.arctan(dy(phi)/dx(phi))")
    problem.add_equation("ep = (vartheta) + (np.cos(theta))")
    problem.add_equation("ep_prime = -np.sin(theta)")
    
    x0, y0 = 0, 0  # Center of the ellipsoid
    radius = 3.0
    phi['g'] = circle_with_noise(x, y, x0, y0, radius)
    theta['g'] = 0.0
    
    return problem, dist, xbasis, ybasis, phi, ep, ep_prime, sig, r, theta, x, y

def paraview_analysis(phi, sig, dist, solver, Lx, Ly, Nx, Ny, save_file):
    phi_full = np.array((phi['g'].copy()))
    sig_full = np.array((sig['g'].copy()))
    domain_phi = phi.domain
    domain_sig = sig.domain
    # Save distributed arrays using the provided function with the full shape
    ut.save_distributed_array(Nx, Ny, domain_phi, dist, phi_full, save_file + '/p' + str(solver.iteration), full_shape=(Nx, Ny), binary_format='bin')
    ut.save_distributed_array(Nx, Ny, domain_sig, dist, sig_full, save_file + '/s' + str(solver.iteration), full_shape=(Nx, Ny), binary_format='bin')


def main(l_f, l_phi, lr_phi, l1_sig, l2_sig, lr_sig, k2, save_file):
    import numpy as np
    Lx, Ly = 4.0, 4.0
    Nx, Ny = 512, 512

    initial_dt = 5.0e-5
    stop_sim_time = 2.0

    dealias = 3/2
    timestepper = d3.RK443
    dtype = np.float64

    problem, dist, xbasis, ybasis, phi, ep, ep_prime, sig, r, theta, x, y = setup_problem(Lx, Ly, Nx, Ny, l_f, l_phi, lr_phi, l1_sig, l2_sig, lr_sig, k2, dealias, dtype)
    initialize_fields(x, y, Lx, Ly, Nx, Ny, phi, ep, ep_prime, sig, r, theta, l_f, l_phi, lr_phi, l1_sig, l2_sig, lr_sig, k2)

    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    n_snaps = 100 # Number of frames generated
    analysis = solver.evaluator.add_file_handler('analysis_apoptosis', sim_dt=stop_sim_time/n_snaps, max_writes=100000)
    
    analysis.add_task(phi, name='phi')
    analysis.add_task(sig, name='sig')

    try:
        logger.info('Starting main loop')
        current_dt = initial_dt
        while solver.proceed:
            if solver.iteration % 1000 == 1:
                paraview_analysis(phi, sig, dist, solver, Lx, Ly, Nx, Ny, save_file)

            solver.step(current_dt)
            if (solver.iteration-1) % 10 == 0:
                output_time = solver.sim_time
                logger.info('Iteration=%i, Time=%e, dt=%e' % (solver.iteration, output_time, current_dt))
    except Exception as e:
        logger.error('Exception details: %s' % str(e))
        raise
    finally:
        solver.log_stats()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AC_global_const_2D with parameters.')
    parser.add_argument('l_f', type=float, help='Parameter l_f')
    parser.add_argument('l_phi', type=float, help='Parameter l_phi')
    parser.add_argument('lr_phi', type=float, help='Parameter lr_phi')
    parser.add_argument('l1_sig', type=float, help='Parameter l1_sig')
    parser.add_argument('l2_sig', type=float, help='Parameter l2_sig')
    parser.add_argument('lr_sig', type=float, help='Parameter lr_sig')
    parser.add_argument('k2', type=float, help='Parameter k2')
    parser.add_argument('save_file', type=str, help='Directory to save analysis files')
    
    args = parser.parse_args()

    main(args.l_f, args.l_phi, args.lr_phi, args.l1_sig, args.l2_sig, args.lr_sig, args.k2, args.save_file)