import numpy as np

import os, sys, csv, time

import gc

path = os.path.dirname(__file__) 
sys.path.append(path)

'''
    - Goal of the code: MRCI of n states
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

'''
    MAIN PART:
'''
in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# 1st: read in two input files specifying parameters and the wavefunction files
print('\nParameters 1st input file:')
params_calc = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

print('\nParameters 2nd input file:')
params_wf_files = in_object.get_parameters_imag_time_prop(path+'/', arg=2)


in_object = h_in.coupl_states(params_calc=params_calc, params_wfs=params_wf_files)
folder_name_res, mrci_energies_file_name = in_object.energy_results_coupling_of_states(path) 

V_0_array = np.array(params_calc['V_0'], dtype=float)
n_states = params_wf_files['n_states']

print('\nNow compute and diagonalize the effective hamiltonian. Wait for results!')

# object to compute the coupling of states
coupl_object = energy.coupling_of_states(params=params_calc)
for i in range(len(V_0_array)):
    tic = time.perf_counter() # start timer

    V_0 = V_0_array[i]
    print('\nV_0: ', V_0)

    psi_arr, q_arr = in_object.get_wavefunctions_per_interaction(path, V_0)

    # compute and diagonalize the hamiltonian
    coupl_object.V_0 = V_0
    h_eff, s_overlap = coupl_object.calc_hamiltonian(n_states, psi_arr, q_arr)
    e_vals, e_kets, s_e_vec = coupl_object.diag_hamiltonian(h_eff, s_overlap)
    
    # compute transition probs
    trans_probs = coupl_object.transition_probabilities(n_states, e_kets, s_overlap)

    # store energies and transition probs
    in_object.store_energies(V_0, e_vals, path)
    in_object.store_transition_probabilities(V_0, trans_probs, path)

    print('Energies =', e_vals)
    print('\nH_eff =', h_eff)
    print('S =', s_overlap)

    toc = time.perf_counter() # end timer
    print("\nExecution time = ", (toc-tic)/60, "min\n")

    del psi_arr
    gc.collect()