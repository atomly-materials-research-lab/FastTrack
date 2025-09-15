from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.bfgs import BFGS
from ase import Atom
from ase.constraints import FixAtoms
from FastTrack.config import Calculator
import ase.io
from ase.filters import FrechetCellFilter
from ase.optimize.fire import FIRE
import numpy as np


class StructureOptimizer:
    """
    Crystal structure optimizer, supporting multiple optimization methods:
    1 Optimize lattice and atomic coordinates.
    2 Optimize only atomic coordinates.
    3 Optimize by fixing atoms and adding new atoms.
    """

    def __init__(self, input_file, output_file='POSCAR-out', fmax=0.02, steps=300):
        """
        :param input_file: 
        :param output_file: 
        :param fmax: 
        :param steps: 
        """
        self.input_file = input_file
        self.output_file = output_file
        self.fmax = fmax
        self.steps = steps

        self.calc = Calculator()

       
        struc = Structure.from_file(input_file)
        adp = AseAtomsAdaptor()
        self.atoms = adp.get_atoms(struc)
        self.atoms.calc = self.calc

    def optimize_structure(self):
        """
        Optimize the crystal structure (including lattice and atomic coordinates).
        """
        optimizer = FrechetCellFilter(self.atoms,)
        FIRE(optimizer,logfile= None).run(fmax=self.fmax, steps=self.steps)
        self._save_structure()

    def optimize_fix(self):
        """
        Optimize crystal structure (only optimize atomic coordinates).
        """
        optimizer = BFGS(self.atoms)
        optimizer.run(fmax=self.fmax, steps=self.steps)
        self._save_structure()

    def optimize_with_fix_and_add_atom(self, atom_type='Li', cart_pos=None):
        """
        Fix all atoms and optimize the structure by adding a new atom.

        :param atom_type:  eg'Li'
        :param cart_pos:  list   [x, y, z]
        """
        if cart_pos is None:
            raise ValueError("The Cartesian coordinates of the new atom must be provided")

        
        indices = list(range(len(self.atoms))) 
        constraint = FixAtoms(indices=indices)  
        self.atoms.set_constraint(constraint)

        self.atoms.append(Atom(atom_type, position=cart_pos))

       
        optimizer = BFGS(self.atoms ,maxstep =0.1)
        optimizer.run(fmax=self.fmax, steps=self.steps)

        
        #self._save_structure() 
        energy = self.atoms.get_potential_energy()

        return    energy   , self.atoms.positions[-1]

    def _save_structure(self):
       
        ase.io.write(self.output_file, self.atoms, 'vasp')



def expand_cell_by_lattice_length(input_file,outputfile, min_length=7):
    """
    Expand the structure to ensure that each lattice vector (a, b, c) is at least greater than min_1ength.
    If a lattice vector is less than min_1ength, expand the cell in that direction.
    
    :param structure: pymatgen  Structure 
    :param min_length:  7 Å
    """
    
    structure = Structure.from_file(input_file)
    a, b, c = structure.lattice.abc
    
    scale_factors = np.ceil(np.array([min_length / a, min_length / b, min_length / c])).astype(int)
    
    scale_factors = np.maximum(scale_factors, 1)
      
    expanded_structure = structure * scale_factors
     
    expanded_structure.to(fmt="poscar", filename=outputfile)

    return scale_factors


def optimize_structure_with_fixed_atom(structure_file, cart_pos, atom_symbol='Li', fmax=0.01, steps=1000,di=3):
    """
    Add an atom to a given structure, fix its position, 
    optimize the positions of surrounding atoms, and return the optimized energy.

    :param structure_file:  
    :param cart_pos:  list  Å
    :param atom_symbol: 'Li'
    :param fmax:  0.01 eV/Å
    :param steps:  1000
    :return: eV
    """
   
    struc = Structure.from_file(structure_file)
    adp = AseAtomsAdaptor()
    atoms = adp.get_atoms(struc)
    atoms.calc = Calculator()

    
    new_atom_index = len(atoms) 
    atoms.append(Atom(atom_symbol, position=cart_pos))  

    indices_to_fix = []
    for i in range(len(atoms)-1):
        dist = atoms.get_distance(new_atom_index, i, mic=True) 
        if dist > di :
            indices_to_fix.append(i)

    indices_to_fix.append(new_atom_index)  
    constraint = FixAtoms(indices=indices_to_fix)
    atoms.set_constraint(constraint)

    optimizer = BFGS(atoms,logfile=None)
    optimizer.run(fmax=fmax, steps=steps) 


    e = atoms.get_potential_energy()
    return e ,atoms