import numpy as np

import os, sys, csv, time

import gc

path = os.path.dirname(__file__) 
sys.path.append(path)

'''
    - Goal of the code: calculation of the coupling between 2! states
    - Call: python3 a6_coupling_n_states.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - Computes an effective hamiltonian and diagonalizes it (=generalized eigenvalue problem, considering the overlap matrix )
        - Storing the result in separate text-file for separate plotting
        - Results are: energies and the eigenvectors

    - Main output: energies from diagonalized hamiltonian, eigenvectors
'''

# import user-defined classes
import class_energy as energy
import class_handle_input as h_in
import class_preparation as prep 
import class_visualization as vis

'''
    MAIN PART:
'''
in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# 1st: read in two input files specifying parameters and the wavefunction files
print('\nParameters 1st input file:')
params_calc = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

print('\n\nParameters 2nd input file:')
params_wf_files = in_object.get_parameters_imag_time_prop(path+'/', arg=2)

V_0_array = np.array(params_calc['V_0'], dtype=float)

in_object = h_in.coupl_states(params_calc=params_calc, params_wfs=params_wf_files)
folder_name_res, mrci_energies_file_name = in_object.energy_results_coupling_of_states(path) 

n_states = params_wf_files['n_states']


print('\nNow compute and diagonalize the effective hamiltonian. Wait for results!')


# object to compute the coupling of the two states - for this, the first input file is important
# IMPORTANT: Mx and My should be the new! Mx and My, after expanding the grid - thus don't forgot to set this in the input file!!!/create a new one in case
coupl_object = energy.coupling_of_states(params=params_calc)

for i in range(len(V_0_array)):
    tic = time.perf_counter() # start timer

    V_0 = V_0_array[i]
    print('\nV_0: ', V_0)

    psi_arr, q_arr = in_object.get_wavefunctions_per_interaction(path, V_0)
    
    print('Read successful! Wait for results!')

    coupl_object.V_0 = V_0
    h_eff, s_overlap = coupl_object.calc_hamiltonian(n_states, psi_arr, q_arr)

    # diagonalize the hamiltonian
    e_vals, e_kets, s_e_vec = coupl_object.diag_hamiltonian(h_eff, s_overlap)
    
    print(e_vals)
    in_object.store_energies(V_0, e_vals, path)

    for j in range(n_states):
        ket_length = np.sqrt(np.conjugate(e_kets[:,j].T)@s_overlap@e_kets[:,j])
        e_kets[:,j] = e_kets[:,j]/ket_length # normalize the eigenvectors with overlap matrix
        
        amp_j = s_overlap@e_kets[:,j] # amplitudes for the j-th state
        trans_prob_j = np.abs(amp_j)**2

        in_object.store_transition_probabilities(i+1, V_0, trans_prob_j, path)

        #for m in range(n_states):
        #    transition_probs[i,j,m] = 
    
    print('Energies =', e_vals)
    print('\nH_eff =', h_eff)
    print('S =', s_overlap)

    toc = time.perf_counter() # end timer
    print("\nExecution time = ", (toc-tic)/60, "min\n")

    del psi_arr
    gc.collect()