# File for running simulation
import subprocess
import numpy as np
from scipy.stats import qmc

def main():
    n_samples = 1 #3

    #         beta,    ebar,    alpha,    k1  k2   gamma tau
    p_sets = [5.0e-00, 2.0e-04, 9.0e-01, 4.0, 1.0, 1.0, 3.0e-04]

    for i in range(n_samples):
        # Command to run Apoptosis.py
        cmd1 = [
            'mpiexec', '-n', '4', 'python3', 'Apoptosis.py',
            str(p_sets[0]), str(p_sets[1]), str(p_sets[2]), str(p_sets[3]), str(p_sets[4]), str(p_sets[5]), str(p_sets[6]) 
        ]

        # Run the first command
        process1 = subprocess.run(cmd1, check=True)
        
if __name__ == "__main__":
    main()
