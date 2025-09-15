from FastTrack.structure_processing import *
from FastTrack.Calculator import *
from FastTrack.visualization import *
from FastTrack.neb_c import *
from FastTrack.Ionremove import *

def kkk(input_file , atom_type, atom_env):
    """
    Args:
        input_file(str): Name of the file to read from  
        atom_type(str): Type of migrate ion
        atom_env(int):  1  one vacancy around the migrating atom 
                        2  two vacancies  around the migrating atom
                        0  all vacancies limit

    Examples:
        kkk("LiCoO2.cif",'Li',1)
    """

    #Supercell Construction
    scale_factors = expand_cell_by_lattice_length(input_file,"POSCAR-expand")
    print(f"** Expansion factor of the supercell (a, b, c): {scale_factors}",flush=True)

    #optimize structure
    print('** optimize structure',flush=True)
    optimizer = StructureOptimizer( "POSCAR-expand", "POSCAR-op", fmax=0.01, steps = 300)
    optimizer.optimize_structure()

    print('** find initial/final migration sites',flush=True)
    env_paths = env_path_finder("POSCAR-op",atom_type)
    max_e_op_s =[]
    for i ,env_path in enumerate(env_paths):
        dir_name = f'path{i+1}'
        os.makedirs(dir_name, exist_ok=True)

        max_e , max_e_op  = hw( "POSCAR-op" , atom_type , atom_env , dir_name , env_path )
        max_e_op_s.append(max_e_op)

    print("** finish, min barrier ----",min(max_e_op_s))

    return min(max_e_op_s)

def hw(input_file , atom_type, atom_env,  dir_name , env_path):
    #Target Atom Removal
    print('** Target Atom Removal',flush=True)
    output_file_rm = os.path.join(dir_name, "POSCAR_rm")

    remover = IonRemover( input_file , atom_type, env_path)     #input_file = "POSCAR-op"
    if atom_env == 1:   
        remover.if_2()
        ion_pos, nearest_ion_pos = remover.remove_two_ions(output_file_rm)
    elif atom_env == 2:  
        ion_pos, nearest_ion_pos = remover.remove_three_ions(output_file_rm)
    elif atom_env == 0:
        output_file_0 =os.path.join(dir_name, "POSCAR_keep_two")
        ion_pos, nearest_ion_pos = remover.keep_two_closest_ions(output_file=output_file_0)
        StructureOptimizer( output_file_0 , output_file_rm, fmax=0.01, steps = 300).optimize_structure()
    else:
        raise ValueError(f"atom_env wrong")
    
    #print("Initial Position:",ion_pos,"Final Position:",nearest_ion_pos,flush=True)

    #Optimized Initial/Final Positions
    output_file_1 =os.path.join(dir_name, 'POSCAR')
    optimizer = StructureOptimizer( output_file_rm ,output_file_1, fmax=0.01, steps=300)
    e1 ,ion_pos_op = optimizer.optimize_with_fix_and_add_atom(atom_type, ion_pos)
    optimizer2 = StructureOptimizer( output_file_rm, output_file_1, fmax=0.01, steps=300)
    e2, nearest_ion_pos_op = optimizer2.optimize_with_fix_and_add_atom(atom_type, nearest_ion_pos)
    min_energy = min(e1,e2)
    print("** min_energy: --",min_energy ,flush=True)
    print("** Optimized Initial Positions:",ion_pos_op,"Optimized Final Positions:",nearest_ion_pos_op, flush=True)

    #Potential Energy Surface (PES) Sampling
    print('** Potential Energy Surface Sampling',flush=True)
    output_potential =os.path.join(dir_name, "potential.csv")
    nearest_ion_pos_wrapped= compute(  ion_pos_op, nearest_ion_pos_op ,
                    input_file = output_file_rm, 
                    atom_type = atom_type ,
                    output = output_potential
                    )

    #Migration Pathway Identification
    print('** Migration Pathway Identification  ' , "Periodic Boundary Condition Termination",nearest_ion_pos_wrapped, flush=True )
 
    output_path =os.path.join(dir_name, "optimized_path.csv")
    out_pic =os.path.join(dir_name, "Migration_Energy.jpg")
    max_e ,path,e0 = Migration_Energy(output_potential,ion_pos_op ,nearest_ion_pos_wrapped, num_points = 31, iteration_= 30, output=output_path,out_pic=out_pic)
    print('** Relaxation Migration Energy' ,flush=True)

    out_pic_op =os.path.join(dir_name, "Migration_Energy_op.svg")
    out_xsf =os.path.join(dir_name, "Migration.xsf")
    max_e_op = Migration_Energy_op( path , atom_type , 2.8 , out_pic_op , output_file_rm , out_xsf,e0)

    print("="*30)
    print(dir_name,"    Fixed-Lattice Migration Barrier: ",max_e,  "Relaxation Migration Barrier: ",max_e_op ,flush=True)
    print("="*30)

    print('** Potential Energy Surface Visualization', flush=True)
    out_html =os.path.join(dir_name, "Energy_Isosurfaces.html")
    potential_draw(  max_e + 0.1 , min_energy ,output_potential,output_path,out_html)

    return  max_e , max_e_op 



def optimize_struc(input_file, output_file= 'POSCAR-out', fmax=0.03, steps=300):
    optimizer = StructureOptimizer( input_file= input_file, output_file= output_file, fmax=fmax, steps=steps)
    optimizer.optimize_structure()

def neb(initial_struc, final_struc, nimages=7, fmax=0.03 , steps=300):
    """
    NEB
    """
    print('optimize structure')
    optimizer1 = StructureOptimizer( initial_struc, "POSCAR_initial", fmax, steps)
    optimizer1.optimize_fix()
    optimizer2 = StructureOptimizer( final_struc, 'POSCAR_final',fmax, steps)
    optimizer2.optimize_fix()

    success = nebrun_c(nimages, fmax)
    if success:
        print("NEB success!")
        traj_plt_c(nimages)

    else:
        print("NEB Optimization not converging")

