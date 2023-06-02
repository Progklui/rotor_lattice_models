import numpy as np
import scipy 

import matplotlib.pyplot as plt
import matplotlib.ticker

import os, sys, csv, time

import gc

path_main = os.path.dirname(__file__) 
sys.path.append(path_main)
#sys.path.append('/home/fkluiben/Documents/phd_ista/software_projects/rotation_1_lemeshko/rotor_lattice_2d_python')

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
import class_equations_of_motion as eom 
import class_energy as energy
import class_handle_input as h_in
import class_preparation as prep 
import class_visualization as vis

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
# 1st: read in two input files specifying parameters and the wavefunction files
print('\nParameters 1st input file:')
n, M, Mx, My, B, tx, ty, potential_points, Vmin, Vmax, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path_main=path_main+'/', arg=1)
print('\n\nParameters 2nd input file:')
potential_points, Vmin, Vmax, n_states, path1, path2, path3, path4 = in_object.get_parameters_coupling_of_states(path_main=path_main+'/', arg=2)

M = int(Mx*My) # safety - for some input scripts M might not be equal to Mx*My
x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid

V_array = np.linspace(Vmin, Vmax, potential_points)

path_array = [path1, path2, path3, path4] # list of all the paths, so far max. 4 states can be done, but generalization is "straightforward"

print('\nNow compute and diagonalize the effective hamiltonian. Wait for results!')

q = np.array([qx,qy])

# object to compute the coupling of the two states - for this, the first input file is important
# IMPORTANT: Mx and My should be the new! Mx and My, after expanding the grid - thus don't forgot to set this in the input file!!!/create a new one in case
coupl_object = energy.coupling_of_states(Mx=Mx, My=My, B=B, V_0=0, tx=tx, ty=ty, n=n, x=x, dt=dt, tol=tol)

e1_arr = np.zeros((len(V_array)))
e2_arr = np.zeros((len(V_array)))

prob_0I = np.zeros((len(V_array)))
prob_0II = np.zeros((len(V_array)))
prob_0III = np.zeros((len(V_array)))

prob_1I = np.zeros((len(V_array)))
prob_1II = np.zeros((len(V_array)))


e_val_arr = np.zeros((len(V_array), n_states))
e_ket_arr = np.zeros((len(V_array), n_states, n_states), dtype=complex)
transition_probs = np.zeros((len(V_array), n_states, n_states), dtype=complex)

for i in range(len(V_array)):
    tic = time.perf_counter() # start timer
    print('\nPotential: ', V_array[i])

    psi_arr = [] # store here the wavefunctions for the potential point V_array[i] for all n_states
    q_arr = []
    for j in range(n_states):
        psi_arr.append(np.load(path_main+'/'+path_array[j])[i].reshape(My, Mx, n))
        q_arr.append(q)
    
    print('Read successful! Wait for results!')

    coupl_object.V_0 = V_array[i]
    h_eff, s_overlap = coupl_object.calc_hamiltonian(n_states, psi_arr, q_arr)

    # diagonalize the hamiltonian
    e_vals, e_kets, s_e_vec = coupl_object.diag_hamiltonian(h_eff.copy(), s_overlap.copy())
    
    for j in range(n_states):
        e_kets[:,j] = e_kets[:,j]/np.sqrt(np.conjugate(e_kets[:,j].T)@s_overlap@e_kets[:,j]) # normalize the eigenvectors with overlap matrix
        
        amp_j = s_overlap@e_kets[:,j] # amplitudes for the j-th state
        for m in range(n_states):
            transition_probs[i,j,m] = np.abs(amp_j[m])**2

    #amp0 = s_overlap@e_kets[:,0] # amplitudes for the ground state
    #amp1 = s_overlap@e_kets[:,1] # amplitudes for the 1st excited state
    #amp2 = s_overlap@e_kets[:,2] # amplitudes for the 2nd excited state

    #prob_0I[i]   = np.abs(amp0[0])**2 # prob. that ground state remains in ground state
    #prob_0II[i]  = np.abs(amp0[1])**2 # prob. that ground state goes to 1st excited state
    #prob_0III[i] = np.abs(amp0[2])**2 # prob. that ground state goes to 2nd excited state

    #prob_1I[i] = np.abs(amp1[0])**2
    #prob_1II[i] = np.abs(amp1[1])**2

    e_val_arr[i] = e_vals
    #e_ket_arr[i] = e_kets

    #e1_arr[i] = h_eff[0,0].real
    #e2_arr[i] = h_eff[1,1].real
    
    print('Energies =', e_vals)
    print('\nH_eff =', h_eff)
    print('S =', s_overlap)

    toc = time.perf_counter() # end timer
    print("\nExecution time = ", (toc-tic)/60, "min\n")

    del psi_arr
    gc.collect()

print('\nPotential =', V_array)
print('\nMRCI energies =', e_val_arr)

# save results to a text file output - plotting and later analysis can thus be easily outsourced
folder_name = '/prepared_wavefunctions/tx_'+str(tx)+'_ty_'+str(ty)+'_M_'+str(M)+'_B_'+str(B)+'/'\
        +'Vmin_'+str(V_array[0])+'_Vmax_'+str(V_array[len(V_array)-1])+'/energies/'
file_name   = 'energies_MRCI_'+str(n_states)+'_states_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_qx_'+str(qx)+'_qy_'+str(qy)+'_Vmin_'+str(V_array[0])+'_Vmax_'+str(V_array[len(V_array)-1])\
        +'_tol_'+str(tol)+'_dt_'+str(dt)

# saves energies in format (E_tot, E_transfer, E_rotational, E_coupling)
np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'.out', np.transpose([V_array, \
        e_val_arr.T[0], e_val_arr.T[1], e_val_arr.T[2]]), delimiter=' ') 

# save results to a text file output - plotting and later analysis can thus be easily outsourced
folder_name = '/prepared_wavefunctions/tx_'+str(tx)+'_ty_'+str(ty)+'_M_'+str(M)+'_B_'+str(B)+'/'\
        +'Vmin_'+str(V_array[0])+'_Vmax_'+str(V_array[len(V_array)-1])+'/wavefunctions/transition_probabilities_MRCI/'

for i in range(n_states):
    file_name   = 'wavefunction_transition_probabilities_state_'+str(i)+'_MRCI_'+str(n_states)+'_total_states_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_qx_'+str(qx)+'_qy_'+str(qy)+'_Vmin_'+str(V_array[0])+'_Vmax_'+str(V_array[len(V_array)-1])\
            +'_tol_'+str(tol)+'_dt_'+str(dt)

    # saves wavefunctions in format (V_array, prob1, prob2, prob3, evtl. prob4)
    np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'.out', np.transpose([V_array, \
            transition_probs.T[i,0], transition_probs.T[i,1], transition_probs.T[i,2]]), delimiter=' ') 


plt.plot(V_array, prob_0I, marker='x', label=r'state 0, 0')
plt.plot(V_array, prob_0II, marker='x', label=r'state 0, 1')
plt.plot(V_array, prob_0III, marker='x', label=r'state 0, 2')

#plt.plot(V_array, prob_1I, marker='x', label=r'state 1, 1')
#plt.plot(V_array, prob_1II, marker='x', label=r'state 1, 2')

plt.legend()
plt.show()