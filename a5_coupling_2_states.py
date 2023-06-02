import numpy as np
import scipy 

import matplotlib.pyplot as plt
import matplotlib.ticker

import os, sys, csv

path_main = os.path.dirname(__file__) 
sys.path.append(path_main)
#sys.path.append('/home/fkluiben/Documents/phd_ista/software_projects/rotation_1_lemeshko/rotor_lattice_2d_python')

'''
    - Goal of the code: calculation of the coupling between 2! states
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
import class_preparation as prep 
import class_visualization as vis

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
n, M, Mx, My, B, tx, ty, potential_points1, Vmin1, Vmax1, Vback1, pot_points_back1, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path_main=path_main+'/', arg=1)

M = int(Mx*My) # safety - for some input scripts M might not be equal to Mx*My

V_0_pool_f1 = np.linspace(Vmin1, Vmax1, potential_points1)
V_0_pool_b1 = np.linspace(Vmax1, Vback1, pot_points_back1)[::-1]

x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid

folder_name1   = 'matrix_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)\
    +'_V1st_'+str(Vmin1)+'_Vjump_'+str(Vmax1)+'_Vnumber_'+str(potential_points1)+'/'

file_name_f   = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward'
psi_rotors_f  = np.load(in_object.get_file_name(path_main+'/', folder_name1, file_name_f)+'.npy')

file_name_b   = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_backward'
psi_rotors_b  = np.load(in_object.get_file_name(path_main+'/', folder_name1, file_name_b)+'.npy')[::-1] # note the inverse here

# Loop over the different analysis possibilities
print_energ = input('\nShow bare energies (y/n)? ')
plot_object = vis.energies(Mx=Mx, My=My, B=B, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)
if print_energ == 'y':
    plot_object.plot_energies_simple_crossing(psi_rotors_f, psi_rotors_b, Vmin1, Vmax1, Vback1, potential_points1, pot_points_back1, path_main=path_main+'/')

print('\nNow compute and diagonalize the effective hamiltonian. Wait for results!')

psi1 = psi_rotors_f.reshape(potential_points1, My, Mx, n).copy()
psi2 = psi_rotors_b.reshape(pot_points_back1, My, Mx, n).copy()

q1 = np.array([qx,qy])
q2 = np.array([qx,qy])


# object to compute the coupling of the two states
coupl_object = energy.coupling_of_states(Mx=Mx, My=My, B=B, V_0=0, tx=tx, ty=ty, n=n, x=x, dt=dt, tol=tol)

n_states = 2

e1_arr = np.zeros((len(psi1)))
e2_arr = np.zeros((len(psi1)))

prob_0I = np.zeros((len(psi1)))
prob_0II = np.zeros((len(psi1)))

prob_1I = np.zeros((len(psi1)))
prob_1II = np.zeros((len(psi1)))

e_val_arr = np.zeros((len(psi1), n_states))
e_ket_arr = np.zeros((len(psi1), n_states, n_states), dtype=complex)

for i in range(len(psi1)):
    psi_arr = [psi1[i], psi2[i]]
    q_arr   = [q1, q2]

    coupl_object.V_0 = V_0_pool_f1[i]
    h_eff, s_overlap = coupl_object.calc_hamiltonian(n_states, psi_arr, q_arr)

    # diagonalize the hamiltonian
    e_vals, e_kets, s_e_vec = coupl_object.diag_hamiltonian(h_eff.copy(), s_overlap.copy())

    # normalize the eigenvectors with overlap matrix
    e_kets[:,0] = e_kets[:,0]/np.sqrt(np.conjugate(e_kets[:,0].T)@s_overlap@e_kets[:,0])
    e_kets[:,1] = e_kets[:,1]/np.sqrt(np.conjugate(e_kets[:,1].T)@s_overlap@e_kets[:,1])

    amp0 = s_overlap@e_kets[:,0] # amplitudes for the ground state
    amp1 = s_overlap@e_kets[:,1] # amplitudes for the excited state

    prob_0I[i] = np.abs(amp0[0])**2 # prob. that ground state remains in ground state
    prob_0II[i] = np.abs(amp0[1])**2 # prob. that ground state goes to excited state

    prob_1I[i] = np.abs(amp1[0])**2
    prob_1II[i] = np.abs(amp1[1])**2

    e_val_arr[i] = e_vals
    e_ket_arr[i] = e_kets

    e1_arr[i] = h_eff[0,0].real
    e2_arr[i] = h_eff[1,1].real


folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_Vmin_'\
    +str(V_0_pool_f1[0])+'_Vmax_'+str(V_0_pool_f1[len(V_0_pool_f1)-1])+'_complete/coupling_of_states/'
file_name   = 'energ_coupling_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_tol_'+str(tol)+'_dt_'+str(dt)        

np.savetxt(in_object.get_file_name(path_main+'/', folder_name, file_name)+'_energies.out', np.transpose([V_0_pool_f1, \
        e1_arr, e2_arr, e_val_arr.T[0], e_val_arr.T[1]]))

np.savetxt(in_object.get_file_name(path_main+'/', folder_name, file_name)+'_probabilities.out', np.transpose([V_0_pool_f1, \
        prob_0I, prob_0II, prob_1I, prob_1II]))

plot_object.plot_resolved_coupling(n_states, e1_arr, e2_arr, e_val_arr, V_0_pool_f1[0], V_0_pool_f1[len(V_0_pool_f1)-1], len(V_0_pool_f1), path_main+'/')
#plot_object.plot_resolved_coupling_state_contributions(n_states, e_ket_arr, V_0_pool_f1[index_left], V_0_pool_f1[index_right], \
#    int(index_right-index_left), path_main+'/')

plt.plot(V_0_pool_f1, prob_0I, marker='x', label=r'state 0, 1')
plt.plot(V_0_pool_f1, prob_0II, marker='x', label=r'state 0, 2')

plt.plot(V_0_pool_f1, prob_1I, marker='x', label=r'state 1, 1')
plt.plot(V_0_pool_f1, prob_1II, marker='x', label=r'state 1, 2')

plt.legend()

folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_Vmin_'+str(V_0_pool_f1[0])+'_Vmax_'+str(V_0_pool_f1[len(V_0_pool_f1)-1])+'_complete/coupling_of_states/'
file_name   = 'energ_coupling_2d_probabilities_qx_'+str(qx)+'_qy_'+str(qy)+'_tol_'+str(tol)+'_dt_'+str(dt)
plt.savefig(in_object.get_file_name(path_main+'/', folder_name, file_name)+'.png', dpi=400)

plt.show()