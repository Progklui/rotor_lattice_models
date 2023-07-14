import numpy as np

import matplotlib.pyplot as plt
import os, sys, csv, time

import gc

path = os.path.dirname(__file__) 
sys.path.append(path)

# import user-defined classes
import class_diag_hamiltonian as diag_heff
import class_energy as energy
import class_mass_size as mass_size
import class_handle_input as h_in
import class_visualization as vis
import class_handle_wavefunctions as h_wavef

def set_phase(psi, n):
    sign = np.sum((1/(n)**0.5)*psi)
    sign = sign/np.abs(sign)
    return np.conjugate(sign)*psi

def get_indices_for_rotor_ij(i, j, index_list_rotor_1, index_list_rotor_2):
    for p in range(j+1,Mx):
        index_arr_rotor_1 = np.array([i,j])
        index_arr_rotor_2 = np.array([i,p])

        index_list_rotor_1.append(index_arr_rotor_1)
        index_list_rotor_2.append(index_arr_rotor_2)

    for k in range(i+1,My):
        for p in range(Mx):
            index_arr_rotor_1 = np.array([i,j])
            index_arr_rotor_2 = np.array([k,p])

            index_list_rotor_1.append(index_arr_rotor_1)
            index_list_rotor_2.append(index_arr_rotor_2)

    return index_list_rotor_1, index_list_rotor_2

def get_double_rotor_excitation_list(My, Mx):
    index_list_rotor_1 = []
    index_list_rotor_2 = []
    for i in range(My):
        for j in range(Mx):
            index_list_rotor_1, index_list_rotor_2 = get_indices_for_rotor_ij(i, j, index_list_rotor_1, index_list_rotor_2)

    return index_list_rotor_1, index_list_rotor_2

in_object = h_in.params(on_cluster=True) # object for handling inputs from command line

# 1st: read in two input files specifying parameters and the wavefunction files
print('\nParameters 1st input file:')
params = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

Mx = params["Mx"]
My = params["My"]

n = params["n"]

tx = params["tx"]
ty = params["ty"]

scale = ty 

exc_number = params["excitation_no"]

path_wavefunction = params['path']
ferro_order = np.load(path+'/'+path_wavefunction)

coupl_object = energy.coupling_of_states(params=params)
diag_object = diag_heff.diagonalization(params=params)

run_number = 3

new_ferro_gs = ferro_order.copy()
overlap_arr = np.zeros(run_number, dtype=complex)
gs_energ_arr = np.zeros(run_number, dtype=complex)
for t in range(run_number):
    new_ferro_gs_next = new_ferro_gs.copy()
    energy_exc_states, psi_exc_states = diag_object.diag_h_eff(new_ferro_gs_next)

    for i in range(My):
        for j in range(Mx):
            new_ferro_gs[i,j] = set_phase(psi_exc_states[i,j,0], n)
            #new_ferro_gs[i,j] = set_phase(new_ferro_gs[i,j], n) #np.sign(np.sum((1/(n)**0.5)*new_ferro_gs[i,j]))*new_ferro_gs[i,j]

    gs_energ_arr[t] = energy_exc_states[0,0,0]
    overlap_arr[t] = coupl_object.calc_overlap(new_ferro_gs, new_ferro_gs_next)
    print('Iter =', t, ', Overlap =', overlap_arr[t])


psi_arr = []
psi_arr.append(new_ferro_gs)

'''
single excitations
'''
for m in range(1,exc_number):
    for i in range(My):
        for j in range(Mx):
            psi = new_ferro_gs.copy()
            psi[i,j] = set_phase(psi_exc_states[i,j,m], n) #np.sign(np.sum((1/(n)**0.5)*psi_exc_states[i,j,m]))*psi_exc_states[i,j,m] # psi_exc_states[i,j,n]

            psi_arr.append(psi)

'''
double excitations
'''
index_list_rotor_1, index_list_rotor_2 = get_double_rotor_excitation_list(My, Mx)
inter_combinations = len(index_list_rotor_1)
for m1 in range(1,exc_number):
    for m2 in range(1,exc_number):
        for i in range(inter_combinations):
            index_rotor_1 = index_list_rotor_1[i]
            index_rotor_2 = index_list_rotor_2[i]

            psi = new_ferro_gs.copy()
            psi[index_rotor_1[0],index_rotor_1[1]] = set_phase(psi_exc_states[index_rotor_1[0],index_rotor_1[1],m1], n) #np.sign(np.sum((1/(n)**0.5)*psi_exc_states[i,j,m]))*psi_exc_states[i,j,m] # psi_exc_states[i,j,n]
            psi[index_rotor_2[0],index_rotor_2[1]] = set_phase(psi_exc_states[index_rotor_2[0],index_rotor_2[1],m2], n) #np.sign(np.sum((1/(n)**0.5)*psi_exc_states[i,j,m]))*psi_exc_states[i,j,m] # psi_exc_states[i,j,n]

            psi_arr.append(psi)

n_states = len(psi_arr)
q_arr = np.zeros((n_states,2), dtype=complex)

print("No. of states =", n_states)

h_eff, s_ove = coupl_object.calc_hamiltonian(n_states, psi_arr, q_arr)
print("Finished calculation of Hamiltonian!")