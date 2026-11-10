import subprocess
import numpy as np

def main():
    n_samples = 1
    p_sets = [1.5e-00, 4.0e-04 * 0.5, 9.0e-01, 0.9, 10.0, 1.0, 3.0e-04]

    for i in range(n_samples):
        cmd1 = [
            'mpiexec', '-n', '4', 'python3', 'Apoptosis_config_mech.py',
            str(p_sets[0]), str(p_sets[1]), str(p_sets[2]), str(p_sets[3]), str(p_sets[4]), str(p_sets[5]), str(p_sets[6]) 
        ]

        process1 = subprocess.run(cmd1, check=True)

if __name__ == "__main__":
    main()
