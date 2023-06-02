import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg # to import the images later on and stacking them together

import os, sys

path_main = os.path.dirname(__file__) 
sys.path.append(path_main)
#sys.path.append('/home/fkluiben/Documents/phd_ista/software_projects/rotation_1_lemeshko/rotor_lattice_2d_python')

'''
    - Goal of the code: print the densities or phases of a configuration
    - Call: python3 a2_configuration_phases.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - Calculation of densities or phases
        - Gives an intuition of the extension of the polaron

    - Problem:
        - For the necessary large lattice sites (therm. limit), it is unreasonable to plot

    - Main output: stores configuration (density or phase)
    - Other: stack different densities/phases together later
'''

# import user-defined classes 
import class_handle_input as h_in
import class_visualization as vis

# plots the configuration (density or phase) at a given potential - then these things can be collected and printed together
def plot_at_potential(path_main, density_or_phase, V1, psi1, V2, psi2):
    # create object for plotting the configurations
    plot_object = vis.configurations(Mx=Mx, My=My, B=B, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol, Vmin=V1[0], Vmax=V1[len(V1)-1])

    scan = input('Which scan (f/b)? ')
    if scan == 'f':
        V_index = input('Enter potential index): ')
        print("Potential selected at V_0 =", V1[int(V_index)]); print(" ")
        if density_or_phase == 'd':
            plot_object.plot_configuration(psi1, V1, int(V_index), "forward", path_main)
        elif density_or_phase == 'p':
            plot_object.plot_phase(psi1, V1, int(V_index), "forward", path_main)
    elif scan == 'b':
        V_index = input('Enter potential index): ')
        print("Potential selected at V_0 =", V2[int(V_index)]); print(" ")
        if density_or_phase == 'd':
            plot_object.plot_configuration(psi2, V2, int(V_index), "forward", path_main)
        elif density_or_phase == 'p':
            plot_object.plot_phase(psi2, V2, int(V_index), "forward", path_main)
    else:
        Exception: print('Specify forward (f) or backward (b)!')

    return V_index, scan

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
n, M, Mx, My, B, tx, ty, potential_points, Vmin, Vmax, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path_main=path_main+'/', arg=1)

M = int(Mx*My) # safety - for some input scripts M might not be equal to Mx*My

V_0_pool_f = np.linspace(Vmin, Vmax, potential_points)
V_0_pool_b = np.linspace(Vmax, Vback, pot_points_back)[::-1] # generate the potential points for backward scan - note that array is reversed (kept for logic)

x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid

folder_name   = 'matrix_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)\
    +'_V1st_'+str(Vmin)+'_Vjump_'+str(Vmax)+'_Vnumber_'+str(potential_points)+'/'

file_name_f   = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward'+'_init_'+str(init)+'_init_repeat_'+str(init_rep)
psi_rotors_f  = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name_f)+'.npy')

file_name_b   = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_backward'+'_init_'+str(init)+'_init_repeat_'+str(init_rep)
psi_rotors_b  = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name_b)+'.npy')[::-1] # important to reverse array!


index_list = []
phase_list = []

print(' ')
density_or_phase = input('Plot density or phase (d/p)? ')

# Loop over the different analysis possibilities
print(' ')
accept = "n"
while accept != "y":
    V_index, phase = plot_at_potential(path_main+'/', density_or_phase, V_0_pool_f, psi_rotors_f, V_0_pool_b, psi_rotors_b)
    if input('Use configuration (y/n)? ') == 'y':
        index_list.append(int(V_index))
        phase_list.append(phase)
    accept = input('Finished selection (y/n)? ')
    print(' ')

fig, axs = plt.subplots(1, len(index_list))
fig.patch.set_visible(False)

if density_or_phase == 'd':
    file_names = 'psi_rotors_2d_configuration_V_0'
elif density_or_phase == 'p':
    file_names = 'psi_rotors_2d_phase_V_0'

j = 0
for i in index_list:
    folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_Vmin_'+str(V_0_pool_f[0])+\
        '_Vmax_'+str(V_0_pool_f[len(V_0_pool_f)-1])+'_complete/configurations/'
    
    if phase_list[j] == 'f':
        if density_or_phase == 'd':
            file_name  = 'psi_rotors_2d_configuration_V_0_'+str(V_0_pool_f[i])+'_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward.png'
        elif density_or_phase == 'p':
            file_name  = 'psi_rotors_2d_phase_V_0_'+str(V_0_pool_f[i])+'_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward.png'
        file_names = file_names+'_'+str(V_0_pool_f[i])
    elif phase_list[j] == 'b':
        if density_or_phase == 'd':
            file_name  = 'psi_rotors_2d_configuration_V_0_'+str(V_0_pool_b[i])+'_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward.png'
        elif density_or_phase == 'p':
            file_name  = 'psi_rotors_2d_phase_V_0_'+str(V_0_pool_b[i])+'_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward.png'
        file_names = file_names+'_'+str(V_0_pool_b[i])
    
    img = mpimg.imread(folder_name+file_name)
    axs[j].imshow(img)

    axs[j].set_xticks([])
    axs[j].set_yticks([])
    axs[j].axis('off')
    j += 1

plt.box(False)
plt.tight_layout()

plt.subplots_adjust(wspace=0, hspace=0)

file_names = file_names+'_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward.png'
plt.savefig(folder_name+file_names, dpi=400, bbox_inches='tight')
plt.show()