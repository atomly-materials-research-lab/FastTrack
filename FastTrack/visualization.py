import numpy as np
import pandas as pd
from ase.io import read, write
import os
from ase.constraints import FixAtoms
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator,make_interp_spline
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from FastTrack.structure_processing import optimize_structure_with_fixed_atom
import matplotlib.ticker as ticker


def optimize_point_in_grid(idx, path,interpolator, start_point, end_point, regularization_coeff=0.1):
    grid_size = 0.6
    grid_points = 50
    
    current_point = path[idx]
    line_direction = (end_point - start_point) / np.linalg.norm(end_point - start_point)  
    plane_normal = line_direction  
    
    u = np.cross(plane_normal, np.array([1, 0, 0]))  
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(plane_normal, np.array([0, 1, 0]))  
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    
   
    grid_range = np.linspace(-grid_size / 2, grid_size / 2, grid_points)
    grid_u, grid_v = np.meshgrid(grid_range, grid_range)
    
    
    grid_points_3d = (
        current_point
        + grid_u[..., None] * u
        + grid_v[..., None] * v
    )
    grid_points_3d = grid_points_3d.reshape(-1, 3)  
    
   
    grid_energies = interpolator(grid_points_3d)
    
    
    prev_point = path[idx - 1] if idx > 0 else current_point 
    next_point = path[idx + 1] if idx < len(path) - 1 else current_point 
    
    regularization_term = regularization_coeff * (
        np.linalg.norm(grid_points_3d - prev_point, axis=1) +
        np.linalg.norm(grid_points_3d - next_point, axis=1)
    )
    
    
    total_energies = grid_energies + regularization_term
    
    
    min_energy_idx = np.argmin(total_energies)
    min_energy_point = grid_points_3d[min_energy_idx]
    
    return min_energy_point


def optimize_point_with_neighbors(idx, path, interpolator, grid_size=0.4, grid_points=50, regularization_coeff=0.1):
    """
    
    """
    prev_point = path[idx - 1]
    next_point = path[idx + 1]
    
    
    midpoint = (prev_point + next_point) / 2
    
    
    line_direction = next_point - prev_point
    line_direction /= np.linalg.norm(line_direction)  
    
    
    u = np.cross(line_direction, np.array([1, 0, 0])) 
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(line_direction, np.array([0, 1, 0]))  
    u /= np.linalg.norm(u)
    v = np.cross(line_direction, u)  
    
    
    grid_range = np.linspace(-grid_size / 2, grid_size / 2, grid_points)
    grid_u, grid_v = np.meshgrid(grid_range, grid_range)
    
   
    grid_points_3d = (
        midpoint
        + grid_u[..., None] * u
        + grid_v[..., None] * v
    )
    grid_points_3d = grid_points_3d.reshape(-1, 3)  
    
    
    grid_energies = interpolator(grid_points_3d)
    
    
    regularization_term = regularization_coeff * (
        np.linalg.norm(grid_points_3d - prev_point, axis=1) +
        np.linalg.norm(grid_points_3d - next_point, axis=1)
    )
    
    
    total_energies = grid_energies + regularization_term
    
    
    min_energy_idx = np.argmin(total_energies)
    min_energy_point = grid_points_3d[min_energy_idx]
    
    return min_energy_point

def Migration_Energy(potential_file, s_point ,e_point, num_points = 11, iteration_=10, output="optimized_path.csv",out_pic="Migration_Energy.jpg"):

    csv_file = potential_file    #"potential.csv"
    df = pd.read_csv(csv_file)

    
    points = df[['x', 'y', 'z']].values  #  (N, 3)
    energy = df['energy'].values  # (N,)

    interpolator = LinearNDInterpolator(points, energy, fill_value=np.max(energy))
    #interpolator = RBFInterpolator(points, energy, kernel='thin_plate_spline')  # 'cubic''thin_plate_spline'

    
    start_point = np.array(s_point)   
    end_point = np.array(e_point)   

    
    num_points = num_points  
    path = np.linspace(start_point, end_point, num_points)


    for iteration in range(100):     
        for i in range(1, num_points - 1):  
            new_point = optimize_point_in_grid(i, path,interpolator , start_point, end_point)
            path[i] = new_point                 
        if iteration > iteration_:
            print(f"Optimization converged after {iteration + 1} iterations.")
            break
    else:
        print("Reached maximum number of iterations without full convergence.")


    # 
    fixed_indices = [0, len(path) // 2, len(path) - 1]
    for iteration in range(100):       #max_iterations
        for i in range(1, num_points - 1):  
            if i not in fixed_indices:  
                path[i] = optimize_point_with_neighbors(i, path, interpolator)    
        
        if iteration > iteration_:
            print(f"Second optimization converged after {iteration + 1} iterations.")
            break
    else:
        print("Reached maximum number of iterations without full convergence in the second optimization.")

    
    output_file = output
    np.savetxt(output_file, path, delimiter=",", header="x,y,z", comments="")
    print(f"Optimized path saved to {output_file}.")

    
    path_energies = interpolator(path)
    energies_min = np.min(path_energies)
    energies_shifted = path_energies - energies_min  

    max_idx = np.argmax(path_energies)
    max_point = path[max_idx]
    print(f"Highest energy point: {max_point}")
    print("E1------",energies_shifted)

    
    plt.plot(range(len(path_energies)), energies_shifted, marker='o')
    plt.xlabel("Path Point Index")
    plt.ylabel("Energy")
    plt.title("Energy Along the Optimized Path")

    #plt.show()
    plt.savefig(out_pic) 
    plt.close()

    return max(energies_shifted) , path,energies_shifted


def Migration_Energy_op(path, atom_type, di, outf , POSCAR_rm ,out_xsf, e0):   
    """
    
    """
    path_energies=[]
    trajectory = []
    for points in path:
        e , a= optimize_structure_with_fixed_atom(POSCAR_rm, points, atom_symbol= atom_type, fmax=0.01, steps=300,di=di)
        path_energies.append(e)        
        trajectory.append(a)

    write(out_xsf, trajectory, format="xsf")

    energies_min = np.min(path_energies)
    energies_shifted = path_energies - energies_min  

    plot_energy_profile_fixed_labels(
        image_indices1 =  np.linspace(0, 1, len(energies_shifted)),
        energies1= e0,
        image_indices2= np.linspace(0, 1, len(energies_shifted)),
        energies2=  energies_shifted ,
        output_filename= outf ,
        save_plot=True,
        show_plot=False
    )

    return  max(energies_shifted)



def potential_draw(energy_b , min_energy,output_potential, output_path , out_html):
    
    df = pd.read_csv(output_potential) 
    df.loc[df['energy'] < (min_energy - 0.1), 'energy'] = min_energy

    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    energy = df['energy'].values

    
    interpolator = LinearNDInterpolator(points=(x, y, z), values=energy)

    
    grid_x, grid_y, grid_z = np.mgrid[
        x.min():x.max():100j,  
        y.min():y.max():100j,
        z.min():z.max():100j,
    ]

   
    grid_energy = interpolator(grid_x, grid_y, grid_z)

    grid_energy = gaussian_filter(grid_energy, sigma=2)

    
    fig = go.Figure()

    
    fig.add_trace(
        go.Isosurface(
            x=grid_x.ravel(),
            y=grid_y.ravel(),
            z=grid_z.ravel(),
            value=grid_energy.ravel(),
            isomin=np.nanmin(grid_energy)-0.1,  
            isomax=np.nanmin(grid_energy)+ energy_b,  
            surface_count=6,  
            colorscale='Viridis',  
            caps=dict(x_show=False, y_show=False, z_show=False), 
            colorbar_title="Energy", 
            opacity=0.7  
        )
    )

    
    loaded_path = np.loadtxt(output_path, delimiter=",", skiprows=1) 

    
    fig.add_trace(
        go.Scatter3d(
            x=loaded_path[:, 0],  
            y=loaded_path[:, 1],  
            z=loaded_path[:, 2],  
            mode='markers+lines', 
            marker=dict(size=5, color='red'),  
            line=dict(color='blue', width=3),  
            name="Optimized Path"  
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        title="3D Energy Isosurfaces with Smooth Interpolation"
    )

    fig.write_html(out_html)


def plot_energy_profile_fixed_labels(
    image_indices1, energies1,
    image_indices2=None, energies2=None,
    output_filename="energy_profile_subplot.svg",
    save_plot=True, show_plot=False
):
    #
    label1_text = "Fixed"
    color1_val = "#4a90d9"  
    marker1_style = "o"

    label2_text = "Optimized"
    color2_val = "#e17c72"  
    marker2_style = "s"

    x_label_text = "Path distance"
    y_label_text = "Energy (eV)"
    #y_limits_val = (0, 0.41)
    num_smooth_points_val = 300
    spline_k_val = 2
    figsize_val = (5, 4)     #(2.5, 2.2) 

    # 
    prominent_label_fontsize = 13
    plt.rcParams.update({
        'font.family': 'sans-serif',
        #'font.sans-serif': ['Arial'],
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.labelcolor': '#444444',  
        'xtick.color': '#444444',
        'ytick.color': '#444444',
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'lines.linewidth': 1.3,
        'lines.markersize': 4,
    })

    fig, ax = plt.subplots(figsize=figsize_val)

    # 
    img_idx1_np = np.asarray(image_indices1)
    en1_np = np.asarray(energies1)

    x_smooth1 = np.linspace(img_idx1_np.min(), img_idx1_np.max(), num_smooth_points_val)
    if len(img_idx1_np) > spline_k_val:
        spl1 = make_interp_spline(img_idx1_np, en1_np, k=spline_k_val)
        y_smooth1 = spl1(x_smooth1)
        ax.plot(x_smooth1, y_smooth1, color=color1_val)
    else:
        ax.plot(img_idx1_np, en1_np, color=color1_val)
    ax.scatter(img_idx1_np, en1_np, marker=marker1_style,
               facecolors='none', edgecolors=color1_val, linewidths=1.0)

    # 
    if image_indices2 is not None and energies2 is not None:
        img_idx2_np = np.asarray(image_indices2)
        en2_np = np.asarray(energies2)

        x_smooth2 = np.linspace(img_idx2_np.min(), img_idx2_np.max(), num_smooth_points_val)
        if len(img_idx2_np) > spline_k_val:
            spl2 = make_interp_spline(img_idx2_np, en2_np, k=spline_k_val)
            y_smooth2 = spl2(x_smooth2)
            ax.plot(x_smooth2, y_smooth2, color=color2_val)
        else:
            ax.plot(img_idx2_np, en2_np, color=color2_val)
        ax.scatter(img_idx2_np, en2_np, marker=marker2_style,
                   facecolors='none', edgecolors=color2_val, linewidths=1.0)

    # 
    ax.set_xlabel(x_label_text)
    ax.set_ylabel(y_label_text)

    #if y_limits_val:
    #    ax.set_ylim(y_limits_val)

    ax.grid(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 
    ax.text(0.5, 0.9, label1_text,
            transform=ax.transAxes,
            color=color1_val,
            fontsize=prominent_label_fontsize,
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))

    if image_indices2 is not None and energies2 is not None:
        ax.text(0.5, 0.1, label2_text,
                transform=ax.transAxes,
                color=color2_val,
                fontsize=prominent_label_fontsize,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))

    
    if save_plot and output_filename:
        plt.tight_layout(pad=0.3)
        plt.savefig(output_filename, format='svg', transparent=True)
        print(f"fig save as: {output_filename}")

    if show_plot:
        plt.tight_layout(pad=0.3)
        plt.show()

    plt.close(fig)