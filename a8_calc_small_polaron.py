import numpy as np

import time
import os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

'''
    - Goal of the code: computes the small polaron wavefunction
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
# object for handling inputs from command line
in_object = h_in.params(on_cluster=False) 
params = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

V_0_array = np.array(params['V_0'], dtype=float)

# initial wavefunction object
wavefunc_object = h_wavef.wavefunctions(params=params)

# energy object
energy_object = energy.energy(params=params) 

# polaron size objects
size_object = mass_size.polaron_size(params=params)
plot_object = vis.configurations(params=params)

# folder structure objects to store results
in_object = h_in.imag_time(params=params)
folder_name_w, file_name_wavefunction = in_object.wavefunction_folder_structure_imag_time_prop(path) 
folder_name_e, file_name_energies = in_object.energy_results_folder_structure_imag_time_prop(path) 
folder_name_p, file_name_size = in_object.polaron_size_results_folder_structure_imag_time_prop(path)

tic = time.perf_counter() # start timer
for V_0 in V_0_array:
    print('Calculating for V_0 =', V_0)

    wavefunc_object.V_0 = V_0
    psi_out = wavefunc_object.create_init_wavefunction('small_polaron')

    # save wavefunction
    np.save(folder_name_w+file_name_wavefunction+str(V_0), psi_out) 

    # save energies
    '''
        TODO: outsource the saving of energies and polaron size here!
    '''
    energy_object.V_0 = V_0
    E = energy_object.calc_energy(psi_out)
    E_analytic = energy_object.analytic_small_polaron_energy()
    with open(folder_name_e+file_name_energies, 'a') as energy_file:
        write_string = str(V_0)+' '+str(E_analytic)+' '+str(E[0])+' '+str(E[1])+' '+str(E[2])+' '+str(E[3])+'\n'
        energy_file.write(write_string)

    # save and plot polaron size
    sigma = size_object.calc_polaron_size(psi_out, '1')
    np.savetxt(folder_name_p+file_name_size+str(V_0)+'.out', (sigma))
    plot_object.plot_polaron_size_imag_time(sigma, V_0, folder_name_p+file_name_size+str(V_0))
    
    del sigma
    del psi_out
    gc.collect()

toc = time.perf_counter() # end timer
print("\nExecution time = ", (toc-tic)/60, "min")