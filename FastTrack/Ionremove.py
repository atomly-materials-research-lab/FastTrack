import random
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.geometry import find_mic
from pymatgen.analysis.diffusion.neb.pathfinder import DistinctPathFinder


class IonRemover_old:
    def __init__(self, input_file, ion_type ):
        """
        :param input_file:Name of the file to read from  
        :param ion_type: Type of migrate ion
        """
        
        self.struc = Structure.from_file(input_file)
        self.adp = AseAtomsAdaptor()
        self.atoms = self.adp.get_atoms(self.struc)
        self.ion_type = ion_type

       
        self.ion_indices = [i for i, atom in enumerate(self.atoms) if atom.symbol == self.ion_type]
        if len(self.ion_indices) < 2:
            raise ValueError(f" {self.ion_type} in struc less then 2")

    def if_2(self):
        target_index = random.choice(self.ion_indices)

        distances_to_A = []
        for i in self.ion_indices:
            if i == target_index:
                continue
            dist = self.atoms.get_distance(target_index, i, mic=True)
            distances_to_A.append((i, dist))
        distances_to_A.sort(key=lambda x: x[1])
        ion_B_index = distances_to_A[0][0]
        di_AB = distances_to_A[0][1]
 
        vec_AB = self.atoms.get_distance(target_index, ion_B_index, 
                                    mic=True, vector=True)
        midpoint = self.atoms[target_index].position + vec_AB/2
 
        midpoint_frac = self.atoms.cell.scaled_positions(midpoint[np.newaxis, :])[0]
        midpoint_frac %= 1.0  
        midpoint = self.atoms.cell.cartesian_positions(midpoint_frac)

        min_dist = float('inf')
        ion_C_index = -1
        for i in self.ion_indices:
            if i in (target_index, ion_B_index):
                continue
            displacement = self.atoms[i].position - midpoint
            mic_disp = find_mic([displacement], 
                            self.atoms.cell, 
                            self.atoms.pbc)[0][0]
            distance = np.linalg.norm(mic_disp)
            if distance < min_dist:
                min_dist = distance
                ion_C_index = i

        di_AC = self.atoms.get_distance(target_index, ion_C_index, mic=True)
        di_BC = self.atoms.get_distance(ion_B_index, ion_C_index, mic=True)
    
        if _check_difference(di_AC, di_BC, di_AB):
            print("---------------------------------------------------")
            print("We suggest setting the atom_env parameter to 2 and recalculating")
            print("---------------------------------------------------")


    def remove_two_ions(self, output_file='POSCAR_rm_two'):
        """
        Remove a target ion and the nearest ion, and return their coordinates

        """
        
        target_index = random.choice(self.ion_indices)
        target_pos = self.atoms[target_index].position
        
        nearest_index, nearest_pos = self._find_nearest_ion(target_index)

        
        del self.atoms[[target_index, nearest_index]]

        
        self._save_structure(output_file)

        return target_pos.tolist(), nearest_pos.tolist()

    def remove_three_ions(self, output_file='POSCAR_rm_three'):
        """
        Remove a target ion and its two nearest ions, 
        and return the coordinates of the target ion and the nearest ion
        """
        
        target_index = random.choice(self.ion_indices)
        target_pos = self.atoms[target_index].position

        
        nearest_indices, nearest_positions = self._find_two_nearest_ions(target_index)
     
        del self.atoms[[target_index, *nearest_indices]]

        self._save_structure(output_file)

        return target_pos.tolist(), nearest_positions[0].tolist()

    def keep_two_closest_ions(self, output_file='POSCAR_keep_two'):
        """
        Randomly select an ion and find the closest ion to it, 
        remove the remaining ions of the same species, 
        and return the retained atomic object
        """
        
        target_index = random.choice(self.ion_indices)
        target_pos = self.atoms[target_index].position
        
        _ , nearest_pos = self._find_nearest_ion(target_index)

        
        ions_to_remove = [i for i in self.ion_indices]
        del self.atoms[ions_to_remove]

        
        self._save_structure(output_file)

       
        return  target_pos , nearest_pos

    def _find_nearest_ion(self, target_index):
        """
        
        """
        distances = []
        for i in self.ion_indices:
            if i == target_index:
                continue  
            dist = self.atoms.get_distance(target_index, i, mic=True)
            distances.append((i, dist))
        nearest_index, _ = min(distances, key=lambda x: x[1])
        nearest_pos = self.atoms[nearest_index].position
        return nearest_index, nearest_pos


    def _find_two_nearest_ions(self, target_index):
        """
        
        """
        
        distances_to_A = []
        for i in self.ion_indices:
            if i == target_index:
                continue
            dist = self.atoms.get_distance(target_index, i, mic=True)
            distances_to_A.append((i, dist))
        distances_to_A.sort(key=lambda x: x[1])
        ion_B_index = distances_to_A[0][0]

        
        vec_AB = self.atoms.get_distance(target_index, ion_B_index, 
                                    mic=True, vector=True)
        midpoint = self.atoms[target_index].position + vec_AB/2

        
        midpoint_frac = self.atoms.cell.scaled_positions(midpoint[np.newaxis, :])[0]
        midpoint_frac %= 1.0  
        midpoint = self.atoms.cell.cartesian_positions(midpoint_frac)

        
        min_dist = float('inf')
        ion_C_index = -1
        
        for i in self.ion_indices:
            if i in (target_index, ion_B_index):
                continue
                
           
            displacement = self.atoms[i].position - midpoint
            mic_disp = find_mic([displacement], 
                            self.atoms.cell, 
                            self.atoms.pbc)[0][0]
            distance = np.linalg.norm(mic_disp)
            
            if distance < min_dist:
                min_dist = distance
                ion_C_index = i

        
        ion_B_pos = self.atoms[ion_B_index].position.copy()
        ion_C_pos = self.atoms[ion_C_index].position.copy()

        return (ion_B_index, ion_C_index), (ion_B_pos, ion_C_pos)



    def _save_structure(self, output_file):
        struc_modified = self.adp.get_structure(self.atoms)
        struc_modified.to(fmt='poscar', filename=output_file)


class IonRemover:
    def __init__(self, input_file, ion_type , env_path):
        """
        :param input_file:Name of the file to read from  
        :param ion_type: Type of migrate ion
        """
        
        self.struc = Structure.from_file(input_file)
        self.adp = AseAtomsAdaptor()
        self.atoms = self.adp.get_atoms(self.struc)
        self.ion_type = ion_type
        self.target_index = env_path[0]
        self.ion_B_index = env_path[1]
       
        self.ion_indices = [i for i, atom in enumerate(self.atoms) if atom.symbol == self.ion_type]
        if len(self.ion_indices) < 2:
            raise ValueError(f" {self.ion_type} in struc less then 2")

    def if_2(self):
        target_index = self.target_index
        ion_B_index = self.ion_B_index
        di_AB = self.atoms.get_distance(target_index, ion_B_index, mic=True)
 
        vec_AB = self.atoms.get_distance(target_index, ion_B_index, 
                                    mic=True, vector=True)
        midpoint = self.atoms[target_index].position + vec_AB/2
 
        midpoint_frac = self.atoms.cell.scaled_positions(midpoint[np.newaxis, :])[0]
        midpoint_frac %= 1.0  
        midpoint = self.atoms.cell.cartesian_positions(midpoint_frac)

        min_dist = float('inf')
        ion_C_index = -1
        for i in self.ion_indices:
            if i in (target_index, ion_B_index):
                continue
            displacement = self.atoms[i].position - midpoint
            mic_disp = find_mic([displacement], 
                            self.atoms.cell, 
                            self.atoms.pbc)[0][0]
            distance = np.linalg.norm(mic_disp)
            if distance < min_dist:
                min_dist = distance
                ion_C_index = i

        di_AC = self.atoms.get_distance(target_index, ion_C_index, mic=True)
        di_BC = self.atoms.get_distance(ion_B_index, ion_C_index, mic=True)
    
        if _check_difference(di_AC, di_BC, di_AB):
            print("---------------------------------------------------")
            print("We suggest setting the atom_env parameter to 2 and recalculating")
            print("---------------------------------------------------")


    def remove_two_ions(self, output_file):
        """
        Remove a target ion and the nearest ion, and return their coordinates

        """
        
        target_pos = self.atoms[self.target_index].position
        nearest_pos = self.atoms[self.ion_B_index].position
        
        del self.atoms[[self.target_index, self.ion_B_index]]
        
        self._save_structure(output_file)

        return target_pos.tolist(), nearest_pos.tolist()

    def remove_three_ions(self, output_file):
        """
        Remove a target ion and its two nearest ions, 
        and return the coordinates of the target ion and the nearest ion
        """
        
        target_pos = self.atoms[self.target_index].position
        nearest_pos = self.atoms[self.ion_B_index].position

        vec_AB = self.atoms.get_distance(self.target_index, self.ion_B_index, 
                                    mic=True, vector=True)
        midpoint = self.atoms[self.target_index].position + vec_AB/2
 
        midpoint_frac = self.atoms.cell.scaled_positions(midpoint[np.newaxis, :])[0]
        midpoint_frac %= 1.0  
        midpoint = self.atoms.cell.cartesian_positions(midpoint_frac)

        min_dist = float('inf')
        ion_C_index = -1
        for i in self.ion_indices:
            if i in (self.target_index, self.ion_B_index):
                continue
            displacement = self.atoms[i].position - midpoint
            mic_disp = find_mic([displacement], 
                            self.atoms.cell, 
                            self.atoms.pbc)[0][0]
            distance = np.linalg.norm(mic_disp)
            if distance < min_dist:
                min_dist = distance
                ion_C_index = i
     
        del self.atoms[[self.target_index,self.ion_B_index , ion_C_index]]

        self._save_structure(output_file)

        return target_pos.tolist(), nearest_pos.tolist()

    def keep_two_closest_ions(self, output_file):
        """
        Randomly select an ion and find the closest ion to it, 
        remove the remaining ions of the same species, 
        and return the retained atomic object
        """
        
        
        target_pos = self.atoms[self.target_index].position
        
        nearest_pos = self.atoms[self.ion_B_index].position

        
        ions_to_remove = [i for i in self.ion_indices]
        del self.atoms[ions_to_remove]

        
        self._save_structure(output_file)

       
        return  target_pos , nearest_pos


    def _save_structure(self, output_file):
        struc_modified = self.adp.get_structure(self.atoms)
        struc_modified.to(fmt='poscar', filename=output_file)


def _check_difference(a, b, c, tolerance=0.10):
    pairs = [(a, b), (a, c), (b, c)]
    for x, y in pairs:
        if max(x, y) == 0:  
            return False
        relative_diff = abs(x - y) / max(x, y)
        if relative_diff > tolerance:
            return False
    return True


def env_path_finder(input_file, ion_type):

    initial_structure = Structure.from_file(input_file)  
    path_finder = DistinctPathFinder(
            structure=initial_structure, 
            migrating_specie = ion_type
        )
    
    all_paths = path_finder.get_paths()
    print(all_paths)

    env_paths = []
    for path in all_paths:
        sites = initial_structure.sites
        isite = path.isite
        esite = path.esite
        site_indices =[ _get_site_idx(sites, isite),
        _get_site_idx(sites, esite)]
        env_paths.append(site_indices)

    return  env_paths


def _get_site_idx(all_sites, spec_site):
    idx= None
    tol =1.e-8
    for i, site in enumerate(all_sites):
        if site.specie == spec_site.specie:
            distance = spec_site.distance(site)
            if distance < tol:
                idx = i
                break
    return idx