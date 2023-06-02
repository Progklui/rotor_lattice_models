import numpy as np
import os, sys

path_main = os.path.dirname(__file__)
sys.path.append(path_main)
#sys.path.append('/home/fkluiben/Documents/phd_ista/software_projects/rotation_1_lemeshko/rotor_lattice_2d_python') 

'''
    - Goal of the code: calculation of the effective mass
    - Call: python3 a4_effective_mass.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - Calculation of effective mass
        - Storing the result in separate text-file for separate plotting
        - Print energies of q=0, qx=1, qy=1, to make a plausibility check

    - Problem:
        - No option to truncate the selected potential points

    - Main output: stores effective masses and plot of them
'''

# import user-defined classes 
import class_handle_input as h_in
import class_visualization as vis
import class_mass_size as mass_size

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
n, M, Mx, My, B, tx, ty, potential_points, Vmin, Vmax, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path_main=path_main+'/', arg=1)

M = int(Mx*My) # for safety - more priority is in the Mx, My variables

V_0_pool = np.linspace(Vmin, Vmax, potential_points)
x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid

folder_name = 'matrix_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_V1st_'+\
    str(Vmin)+'_Vjump_'+str(Vmax)+'_Vnumber_'+str(potential_points)+'/'


# Allow the computation of the effective mass for the forward and backward scan
choice_scan_dir = input('\nWhich scan direction to choose (f/b)? ')
if choice_scan_dir == 'f':
    choice_scan_dir = 'forward'
elif choice_scan_dir == 'b':
    choice_scan_dir = 'backward'
    V_0_pool = V_0_pool[::-1]
    
# part I: q_x = q_y = 0
file_name    = 'psi_rotors_2d_qx_'+str(0.0)+'_qy_'+str(0.0)+'_scan_direction_' + choice_scan_dir
psi_rotors_f = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name)+'.npy').reshape(potential_points, My, Mx, n)

# part II: q_x = 1, q_y = 0
file_name       = 'psi_rotors_2d_qx_'+str(1.0)+'_qy_'+str(0.0)+'_scan_direction_' + choice_scan_dir
psi_rotors_f_qx = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name)+'.npy').reshape(potential_points, My, Mx, n)

# part IV: q_x = 0, q_y = 1
file_name       = 'psi_rotors_2d_qx_'+str(0.0)+'_qy_'+str(1.0)+'_scan_direction_' + choice_scan_dir
psi_rotors_f_qy = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name)+'.npy').reshape(potential_points, My, Mx, n)

mass_object = mass_size.eff_mass(Mx=Mx, My=My, B=B, V_0=V_0_pool, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)

# energies are returned to easily verify a part of the result/plausibility check
mx_eff, mx_0, E_col_1, E_col_qx = mass_object.calc_eff_mass(psi_rotors_f, psi_rotors_f_qx, psi_rotors_f_qx, 1, 0) # m_x
my_eff, my_0, E_col_2, E_col_qy = mass_object.calc_eff_mass(psi_rotors_f, psi_rotors_f_qy, psi_rotors_f_qy, 0, 1) # m_y

plot_object = vis.eff_mass(Mx=Mx, My=My, B=B, V_0=0, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)

# Plot effective mass and also the energies (for double checking!)
print_energ = input('\nShow eff. mass (y/n)? ')
if print_energ == 'y': 
    # for convenience - to have an overview already in the command line
    print('\n mx = ', mx_eff/mx_0[np.newaxis])
    print('\n my = ', my_eff/my_0[np.newaxis])

    plot_object.plot_effective_masses(V_0_pool, mx_eff/mx_0[np.newaxis], my_eff/my_0[np.newaxis], 'log', choice_scan_dir, path_main+'/')

    # store energies in designated folder of effective masses
    folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)\
        +'_Vmin_'+str(np.min(V_0_pool))+'_Vmax_'+str(np.max(V_0_pool))+'_complete/effective_mass/'
    file_name   = 'eff_mass_2d_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_tol_'+str(tol)+'_dt_'+str(dt)+'_'+choice_scan_dir

    np.savetxt(in_object.get_file_name(path_main+'/', folder_name, file_name)+'_mx_my.out', np.transpose([V_0_pool, mx_eff, np.ones(len(mx_eff))*mx_0, my_eff, np.ones(len(my_eff))*my_0])) 

    
print_energ = input('\nShow energies (y/n)? ')
if print_energ == 'y':
    plot_object.plot_eff_mass_energies(V_0_pool, E_col_1, E_col_qx, E_col_qy, choice_scan_dir, path_main+'/')

    # store energies in designated folder of effective masses
    folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)\
        +'_Vmin_'+str(np.min(V_0_pool))+'_Vmax_'+str(np.max(V_0_pool))+'_complete/effective_mass/'
    file_name   = 'energ_2d_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_different_q_tol_'+str(tol)+'_dt_'+str(dt)+'_'+choice_scan_dir

    np.savetxt(in_object.get_file_name(path_main+'/', folder_name, file_name)+'.out', np.transpose([V_0_pool, E_col_1, E_col_qx, E_col_qy]))
