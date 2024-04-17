import numpy as np

import time
import os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

'''
    - Goal of the code: increase the grid size of a given wavefunction
    - Call: python3 a5_coupling_2_states.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - Computes an effective hamiltonian and diagonalizes it (=generalized eigenvalue problem, considering the overlap matrix )
        - Storing the result in separate text-file for separate plotting
        - Results are: energies and the eigenvectors

    - Main output: energies from diagonalized hamiltonian, eigenvectors
'''

# import user-defined classes
import class_equations_of_motion as eom 
import class_energy as energy

import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

import class_visualization as vis
import class_mass_size as mass_size

'''
    MAIN PART:
'''
in_object = h_in.params(on_cluster=True) # object for handling inputs from command line
params = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

V_0_array = np.array(params['V_0'], dtype=float)

Mx_new_list = np.array(params['Mx_new'], dtype=int) 
My_new_list = np.array(params['My_new'], dtype=int)

params_new = params.copy()
params_new['Mx'] = Mx_new_list[-1] 
params_new['My'] = My_new_list[-1] 

# energy objects
energy_object = energy.energy(params=params)
energy_object_new = energy.energy(params=params_new)

# polaron size objects
size_object = mass_size.polaron_size(params=params_new)
plot_object = vis.configurations(params=params_new)

# folder structure objects to read old w.f. and store properties of new w.f.
in_object = h_in.imag_time(params=params)
folder_name_w, file_name_wavefunction = in_object.wavefunction_folder_structure_imag_time_prop(path) 

in_object_new = h_in.imag_time(params=params_new)
folder_name_w_new, file_name_wavefunction_new = in_object_new.wavefunction_folder_structure_imag_time_prop(path)
folder_name_e, file_name_energies = in_object_new.energy_results_folder_structure_imag_time_prop(path) 
folder_name_p, file_name_size = in_object_new.polaron_size_results_folder_structure_imag_time_prop(path)

for V_0 in V_0_array:
    psi = np.load(folder_name_w+file_name_wavefunction+str(V_0)+'.npy')

    if params['converge_new_lattice'] == "yes":
        wavefunc_object = h_wavef.wavefunc_operations(params=params)
        psi_new = wavefunc_object.expand_and_converge_wf(Mx_new_list, My_new_list, V_0, psi)

    elif params['converge_new_lattice'] == "no":
        params_no_imag = params.copy()
        params_no_imag['Mx_new'] = Mx_new_list[-1] 
        params_no_imag['My_new'] = My_new_list[-1] 

        wavefunc_object = h_wavef.wavefunc_operations(params=params_no_imag)
        psi_new = wavefunc_object.add_rotors_to_wavefunction(psi)   

    # compute energies
    energy_object.V_0 = V_0
    E_old = energy_object.calc_energy(psi)

    energy_object_new.V_0 = V_0
    E_new = energy_object_new.calc_energy(psi_new)

    E_diff = E_new - E_old
    print('V_0 =', V_0, ', E_tot (old) =', E_old[0], ', E_tot (new) =', E_new[0], ', E_new - E_old =', E_diff[0])

    # save new wavefunction
    np.save(folder_name_w_new+file_name_wavefunction_new+str(V_0), psi_new) 

    # save new energies
    with open(folder_name_e+file_name_energies, 'a') as energy_file:
        write_string = str(V_0)+' '+str(E_new[0])+' '+str(E_new[1])+' '+str(E_new[2])+' '+str(E_new[3])+'\n'
        energy_file.write(write_string)

    # save and plot new polaron size
    sigma = size_object.calc_polaron_size(psi_new, '1')
    np.savetxt(folder_name_p+file_name_size+str(V_0)+'.out', (sigma))
    plot_object.plot_polaron_size_imag_time(sigma, V_0, folder_name_p+file_name_size+str(V_0))
    
    del sigma
    del psi_new
    gc.collect()