import argparse
import h5py
import matplotlib
from datetime import datetime
matplotlib.use('Agg')  # Use the 'Agg' backend for headless plotting
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools
import pathlib
from dedalus.tools import post
from dedalus.tools.parallel import Sync

def plot_2D(filename, start, count, output):
        """Save plot of specified task for given range of analysis writes."""

        # Plot settings
        tasks = ['phi', 'sig'] 
        scale = 1  # Adjust if necessary for plot size
        dpi = 200  # Dots per inch for output images
        title_func = lambda sim_time: r'$t = {:.3f}$'.format(sim_time)
        # Layout settings
        nrows, ncols = 1, 2  # Adjust based on how many fields you want to plot side by side
        image = plot_tools.Box(1, 1)  # Size of the plot image
        pad = plot_tools.Frame(0.2, 0, 0, 0)  # Padding between plots
        margin = plot_tools.Frame(0.2, 0.1, 0, 0)  # Margin around the image

        # Create multifigure
        mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
        fig = mfig.figure

        # Plot writes
        with h5py.File(filename, mode='r') as file:
            for index in range(start, start + count):
                for n, task in enumerate(tasks):
                    # Build subfigure axes
                    i, j = divmod(n, ncols)
                    axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                    # Call 3D plotting helper, slicing in time
                    dset = file['tasks'][task]
                    plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=r'$\varphi (x,y)$' if task == 'p' else r'$\sigma (x,y)$', even_scale=True, visible_axes=False, clim=(-1.0, 1.0))
                
                # Add time title below subplots
                title = title_func(file['scales/sim_time'][index])
                fig.text(0.5, -0.05, title, ha='center', va='top', transform=fig.transFigure)
                # Save figure
                savename = f"write_{index:06}.png"
                savepath = output / savename
                fig.savefig(str(savepath), dpi=dpi, bbox_inches='tight')
                fig.clear()
        plt.close(fig)

def main(files, output_dir, start, count, l_f, l_phi, lr_phi, l1_sig, l2_sig, lr_sig, k2):
    """
    Plot planes from joint analysis files for Allen-Cahn simulations and save the plots with specified values in the filename.

    Parameters:
        files (list): List of HDF5 files to plot.
        output_dir (str): Output directory to save the plots.
        start (int): Start index for the range of analysis writes.
        count (int): Number of analysis writes to plot.
        ep_p (float): Value for ep_p to include in the filename.
        ep_s (float): Value for ep_s to include in the filename.
        bp (float): Value for bp to include in the filename.
        bs (float): Value for bs to include in the filename.
        kps (float): Value for kps to include in the filename.
    """

    output_path = pathlib.Path(output_dir).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
    post.visit_writes(files, plot_2D, output=output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AC_global_const_2D with parameters.')

    parser.add_argument("file_pattern", type=str)
    parser.add_argument('l_f', type=float, help='Parameter l_f')
    parser.add_argument('l_phi', type=float, help='Parameter l_phi')
    parser.add_argument('lr_phi', type=float, help='Parameter lr_phi')
    parser.add_argument('l1_sig', type=float, help='Parameter l1_sig')
    parser.add_argument('l2_sig', type=float, help='Parameter l2_sig')
    parser.add_argument('lr_sig', type=float, help='Parameter lr_sig')
    parser.add_argument('k2', type=float, help='Parameter k2')
        
    args = parser.parse_args()

    files = list(pathlib.Path().glob(args.file_pattern))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    values_str = f"l_f_{args.l_f}_l_phi_{args.l_phi}_lr_phi_{args.lr_phi}_l1_sig_{args.l1_sig}_l2_sig_{args.l2_sig}_lr_sig_{args.lr_sig}_k2_{args.k2}_{timestamp}"
    output_dir = values_str  # Directory named based on parameter values and timestamp

    start = 0  # Adjust as needed
    count = 10  # Adjust as needed

    main(files, output_dir, start, count, args.l_f, args.l_phi, args.lr_phi, args.l1_sig, args.l2_sig, args.lr_sig, args.k2)