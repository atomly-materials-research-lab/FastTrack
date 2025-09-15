from ase import io
from ase.mep import NEB  
from ase.optimize import MDMin ,BFGS, FIRE
from ase.io.trajectory import Trajectory
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from FastTrack.config import Calculator
from matplotlib.ticker import MaxNLocator, FuncFormatter

def nebrun_c( nimages=5, fmax=0.04):
    """
     NEB by MLFF

    Args:
        poscar_initial (str): 
        poscar_final (str): 
        num_images (int): 
        fmax (float):

    Returns:
        bool: if success
    """
 
    initial = io.read("POSCAR_initial")
    final = io.read("POSCAR_final")

  
    images = [initial]
    images += [initial.copy() for _ in range(nimages)]
    images += [final]

    neb = NEB(images , k=0.05 , climb=True )    #k  Spring constant(s) in eV/Ang.
    #neb = DyNEB(images, fmax, dynamic_relaxation=True, scale_fmax=1.)

    neb.interpolate('idpp')

    for image in images:
        image.calc =  Calculator()

    #optimizer = MDMin(neb, trajectory='A2B.traj')
    #optimizer = BFGS(neb, trajectory='A2B.traj')
    optimizer = FIRE(neb, trajectory='A2B.traj')

    optimization_success = optimizer.run(fmax=fmax)

    return optimization_success


def traj_plt_c (nimages):
    traj = Trajectory('./A2B.traj')
    last_frames = traj[-(nimages+2):]
    io.write('Trajectory.xsf',last_frames,'xsf')

    calculator = Calculator() 
    energies = []
    for atoms in last_frames:
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        energies.append(energy)

    energies = np.array(energies)
    energies_min = np.min(energies)
    energies_shifted = energies - energies_min

    image_indices = np.arange(len(energies))

   
    x_smooth = np.linspace(image_indices.min(), image_indices.max(), 300)
    spl = make_interp_spline(image_indices, energies_shifted, k=2)
    y_smooth = spl(x_smooth)

    plt.plot(x_smooth, y_smooth, color='b', )
    plt.scatter(image_indices, energies_shifted, color='red', label='Original Data Points')
    plt.xlabel("reaction coordinate")
    plt.ylabel("Energy (eV)")
    plt.title("Migration Energy Barrier")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=False))  
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    plt.legend()
    plt.grid(True)
    plt.savefig("MEB.png")  
