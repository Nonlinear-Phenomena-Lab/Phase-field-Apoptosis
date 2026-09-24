# File for running simulation
import subprocess
import numpy as np
import argparse
from scipy.stats import qmc

def main(save_file):
    n_samples = 1 

    l_f = 3.33e+3
    l_phi = 1.333e-04
    lr_phi = 960.0
    l1_sig = 1.0
    l2_sig = 1.5
    lr_sig = 0.28
    k2 = 8.0
    p_sets = [l_f, l_phi, lr_phi, l1_sig, l2_sig, lr_sig, k2]

    for i in range(n_samples):
        # Command to run Apoptosis.py
        cmd1 = [
            'mpiexec', '-n', '8', 'python3', 'ApoptosisNondim.py',
            str(p_sets[0]), str(p_sets[1]), str(p_sets[2]), str(p_sets[3]), str(p_sets[4]), str(p_sets[5]), str(p_sets[6]), save_file 
        ]
        file_pattern = "analysis_apoptosis/analysis_apoptosis_s1/*.h5" 

        cmd2 = [
            'mpiexec', '-n', '1', 'python3', 'plot_run.py', 'analysis_apoptosis/*.h5',
            str(p_sets[0]), str(p_sets[1]), str(p_sets[2]), str(p_sets[3]), str(p_sets[4]), str(p_sets[5]), str(p_sets[6])
        ]

        # Run the first command
        process1 = subprocess.run(cmd1, check=True)

        # process2 = subprocess.run(cmd2, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Cell fusion with parameters.')
    parser.add_argument('save_file', type=str, help='Directory to save analysis files')
    args = parser.parse_args()
    
    main(args.save_file)
