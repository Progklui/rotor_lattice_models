import numpy as np
import os, sys

path_main = os.path.dirname(__file__) 
sys.path.append(path_main)
#sys.path.append('/home/fkluiben/Documents/phd_ista/software_projects/rotation_1_lemeshko/rotor_lattice_2d_python')

'''
    - Goal of the code: calculate energies of a scan
    - Call: python3 a1_energy_gs.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - Calculation of energies and storing in .txt file for individual processing
        - Simple plotting options for quick analysis/overview
        - Plotting for posters, ... should be 'outsourced'

    - Main output: stores energies in folder: image_results/psi_rotors_...parameters.../energies/
    - Other: energies around a potential phase transition
'''

# import user-defined classes 
import class_energy as energy
import class_handle_input as h_in # handles interaction with command line - reading of input script
import class_preparation as prep # used for the analysis of a phase transition
import class_visualization as vis # for simple/standardized plotting operations

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
# First: read in all parameters from input script provided in command line
n, M, Mx, My, B, tx, ty, potential_points, Vmin, Vmax, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path_main=path_main+'/', arg=1)

M = int(Mx*My) # safety - for some input scripts M might not be equal to Mx*My

V_0_pool_f = np.linspace(Vmin, Vmax, potential_points) # generate the potential points for forward scan
V_0_pool_b = np.linspace(Vmax, Vback, pot_points_back)[::-1] # generate the potential points for backward scan - note that array is reversed (kept for logic)

x = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid

folder_name   = 'matrix_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)\
    +'_V1st_'+str(Vmin)+'_Vjump_'+str(Vmax)+'_Vnumber_'+str(potential_points)+'/' # the results for both scans are stored in the same folder

file_name_f   = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward'+'_init_'+str(init)+'_init_repeat_'+str(init_rep)
psi_rotors_f  = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name_f)+'.npy')

file_name_b   = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_backward'+'_init_'+str(init)+'_init_repeat_'+str(init_rep)
psi_rotors_b  = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name_b)+'.npy')[::-1] # important to reverse the array

# Plot and print energies - is an important analysis option
print_energ = input('\nShow bare energies (y/n)? ')
if print_energ == 'y': 
    energy_object = energy.energy(Mx=Mx, My=My, B=B, V_0=0, tx=tx, ty=ty,
                                  qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol) # create energy object 

    E_col_f = np.zeros((len(V_0_pool_f),4), dtype=complex)
    E_col_b = np.zeros((len(V_0_pool_b),4), dtype=complex)

    # compute energies from forward scan
    for i in range(len(V_0_pool_f)):
        energy_object.V_0 = V_0_pool_f[i] # update coupling in energy object
        E_col_f[i] = np.asarray(energy_object.calc_energy(psi_rotors_f[i]))

    # compute energies from backward scan
    for i in range(len(V_0_pool_b)):
        energy_object.V_0 = V_0_pool_b[i] # update coupling in energy object
        E_col_b[i] = np.asarray(energy_object.calc_energy(psi_rotors_b[i])) # [0] was here before?

    print('\nV_0 =', V_0_pool_f)
    print('E_f =', E_col_f.T[0])

    print('\nV_0 =', V_0_pool_b)
    print('E_b =', E_col_b.T[0], '\n')

    # save results to a text file output - plotting and later analysis can thus be easily outsourced
    folder_name = '/image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)\
            +'_Vmin_'+str(V_0_pool_f[0])+'_Vmax_'+str(V_0_pool_f[len(V_0_pool_f)-1])+'_complete/energies/'
    file_name   = 'energ_2d_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_qx_'+str(qx)+'_qy_'+str(qy)\
            +'_tol_'+str(tol)+'_dt_'+str(dt)

    # saves energies in format (E_tot, E_transfer, E_rotational, E_coupling)
    np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_forward_scan.out', np.transpose([V_0_pool_f, \
        E_col_f.T[0], E_col_f.T[1], E_col_f.T[2], E_col_f.T[3]])) 
    np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_backward_scan.out', np.transpose([V_0_pool_b, \
        E_col_b.T[0], E_col_b.T[1], E_col_b.T[2], E_col_b.T[3]]))

    plot_object = vis.energies(Mx=Mx, My=My, B=B, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)
    plot_object.plot_e_simple_crossing(V_0_pool_f, E_col_f.T[0].real, V_0_pool_b, E_col_b.T[0].real, Vmin, Vmax, path_main=path_main+'/')

# Show phase transition (if there is one, otherwise produces an error!) - for convenience and fast analysis
# Although for later plottings (e.g. poster, ...) it is better to write a separate function
print_energ = input('\nShow phase transition (y/n)? ')
if print_energ == 'y':
    print('Comment: just sensible if there really is one! Check energies before!\n')
    
    # first, prepare an object that allows to extract the wave functions for the different phases
    prep_object = prep.phase_transition(psi_rotors_f1=psi_rotors_f, psi_rotors_b1=psi_rotors_b, psi_rotors_f2=None, psi_rotors_b2=None, 
                                        potential_points1=potential_points, potential_points2=None, pot_points_back1=pot_points_back, 
                                        pot_points_back2=None, Mx=Mx, My=My, B=B, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)
    
    V1, V2, psi_phase1, psi_phase2 = prep_object.get_wave_function_indices_2_phases(V_0_pool_f, V_0_pool_b)
    
    V = [V1,V2]
    psi = [psi_phase1, psi_phase2]

    plot_object = vis.energies(Mx=Mx, My=My, B=B, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)
    plot_object.plot_phase_transition(V, psi, path_main=path_main+'/')