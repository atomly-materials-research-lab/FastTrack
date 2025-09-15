import csv
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atom
from FastTrack.config import Calculator
import numpy as np
import concurrent.futures

def compute_energy(i, j, k,xmin, xmax,ymin, ymax, zmin, zmax,  nx,ny,nz, atoms ,atom_type):
    """
    Calculate the energy of a single point and return the result
    """
    z = zmin + (k * (zmax - zmin) / nz)
    x = xmin + (i * (xmax - xmin) / nx)
    y = ymin + (j * (ymax - ymin) / ny)
    # cart_pos
    cart_pos = [x, y, z]
    #cart_pos = frac_pos @ atoms.cell  

    #Remove objects with atoms too close together
    distances = np.linalg.norm(atoms.positions - cart_pos, axis=1)
    if np.min(distances) < 0.5:
        return [cart_pos[0], cart_pos[1], cart_pos[2], 0]


    atoms.calc = Calculator()
      
    atoms.append(Atom(atom_type, position=cart_pos))  
    
    e = atoms.get_potential_energy()
    energy = e #.item()
   
    if energy < 0:
        return [cart_pos[0], cart_pos[1], cart_pos[2], energy]
    else:
        return [cart_pos[0], cart_pos[1], cart_pos[2], 0]


def parallel_compute( xmin, xmax,ymin, ymax,zmin, zmax, nx,ny, nz, input_file,workers,atom_type):
    
    struc = Structure.from_file(input_file)  
    adp = AseAtomsAdaptor()
    atoms = adp.get_atoms(struc)

    with open("potential.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z", "energy"])

        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for k in range(nz):
                for i in range(nx):
                    for j in range(ny):
                        futures.append(executor.submit(compute_energy, i, j, k, xmin, xmax,ymin, ymax,zmin, zmax, nx,ny,nz, atoms,atom_type))

           
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                writer.writerow(result)

def compute(ion_pos, nearest_ion_pos, input_file, atom_type,output):
    
    struc = Structure.from_file(input_file)  
    adp = AseAtomsAdaptor()
    atoms = adp.get_atoms(struc)

    xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz ,nearest_ion_pos_wrapped= _get_wrapping_box(atoms, ion_pos, nearest_ion_pos)

    atoms.calc = Calculator()
    results = []


    for k in range(nz):
        # print(k,"in",nz, flush=True)
        for i in range(nx):
            for j in range(ny):
                
                z = zmin + (k * (zmax - zmin) / nz)
                x = xmin + (i * (xmax - xmin) / nx)
                y = ymin + (j * (ymax - ymin) / ny)
                cart_pos = [x, y, z]

                # Remove objects with atoms too close together
                distances = np.linalg.norm(atoms.positions - cart_pos, axis=1)
                if np.min(distances) < 0.5:
                    results.append([x, y, z, 0]) 
                else:
                    atoms.append(Atom(atom_type, position=cart_pos))
                    e = atoms.get_potential_energy()  
                    energy = e 
                    atoms.pop()

                    if energy < 0:
                        results.append([x, y, z, energy])
                    else:
                        results.append([x, y, z, 0])

    with open(output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)   
        writer.writerow(["x", "y", "z", "energy"])
        writer.writerows(results)
    
    return nearest_ion_pos_wrapped

def _get_wrapping_box(atoms, ion_pos, nearest_ion_pos, padding=1.3):
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    ion_frac = np.linalg.solve(cell.T, ion_pos)
    nearest_ion_frac = np.linalg.solve(cell.T, nearest_ion_pos)

    
    for i in range(3):
        if pbc[i]:
            delta = nearest_ion_frac[i] - ion_frac[i]
            if delta > 0.5:
                nearest_ion_frac[i] -= 1.0
            elif delta < -0.5:
                nearest_ion_frac[i] += 1.0

    
    nearest_ion_pos_wrapped = np.dot(nearest_ion_frac, cell)
    
    xmin = min(ion_pos[0], nearest_ion_pos_wrapped[0]) - padding
    xmax = max(ion_pos[0], nearest_ion_pos_wrapped[0]) + padding
    ymin = min(ion_pos[1], nearest_ion_pos_wrapped[1]) - padding
    ymax = max(ion_pos[1], nearest_ion_pos_wrapped[1]) + padding
    zmin = min(ion_pos[2], nearest_ion_pos_wrapped[2]) - padding
    zmax = max(ion_pos[2], nearest_ion_pos_wrapped[2]) + padding
    nx = 12+ int(abs(ion_pos[0]-nearest_ion_pos_wrapped[0]) * 5)
    ny = 12+ int(abs(ion_pos[1]-nearest_ion_pos_wrapped[1]) * 5)
    nz = 12+ int(abs(ion_pos[2]-nearest_ion_pos_wrapped[2]) * 5)

    return xmin, xmax, ymin, ymax, zmin, zmax , nx , ny ,nz ,nearest_ion_pos_wrapped
