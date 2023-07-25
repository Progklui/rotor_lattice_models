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

def print_number_of_states(calc_n_states, params):
    single_exc_number = (params['excitation_no']-1)*params['My']*params['My']
    double_exc_number = (params['excitation_no']-1)**2*(params['My']*params['My']*(params['My']*params['My']-1)/2)
    print("No. of states =", calc_n_states, \
        ", No. in theory = 4*(1 +", single_exc_number, "+", double_exc_number, ") =", 4*(1+single_exc_number+double_exc_number))

def print_ref_energies(coupl_object, q, ferro_order, ferro_domain_v, ferro_domain_h, small_polaron):
    e_order = coupl_object.calc_hamiltonian_matrix_element(ferro_order, q, ferro_order, q)[0]
    e_d_v = coupl_object.calc_hamiltonian_matrix_element(ferro_domain_v, q, ferro_domain_v, q)[0]
    e_d_h = coupl_object.calc_hamiltonian_matrix_element(ferro_domain_h, q, ferro_domain_h, q)[0]
    e_s = coupl_object.calc_hamiltonian_matrix_element(small_polaron, q, small_polaron, q)[0]

    print('\nE (Ferro Order)      =', e_order)
    print('E (Ferro Domain Vert.) =', e_d_v)
    print('E (Ferro Domain Hor.)  =', e_d_h)
    print('E (Small Polaron)      = ', e_s, '\n')

def plot_ref_densities(params, ferro_order, ferro_domain_v, ferro_domain_h, small_polaron):
    polaron_size_object = mass_size.polaron_size(params=params)

    ferro_order_sigma = polaron_size_object.calc_polaron_size(ferro_order, '1')
    ferro_domain_v_sigma = polaron_size_object.calc_polaron_size(ferro_domain_v, '1')
    ferro_domain_h_sigma = polaron_size_object.calc_polaron_size(ferro_domain_h, '1')
    small_polaron_sigma = polaron_size_object.calc_polaron_size(small_polaron, '1')

    plt.pcolormesh(ferro_order_sigma)
    plt.show()
    plt.pcolormesh(ferro_domain_v_sigma)
    plt.show()
    plt.pcolormesh(ferro_domain_h_sigma)
    plt.show()
    plt.pcolormesh(small_polaron_sigma)
    plt.show()
    
in_object = h_in.params(on_cluster=True) # object for handling inputs from command line
print('\nParameters input file:')
params = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

'''
load wavefunctions
'''
path_wavefunction_1 = params['path1']
path_wavefunction_2 = params['path2']
path_wavefunction_3 = params['path3']
path_wavefunction_4 = params['path4']

ferro_order    = np.load(path+'/'+path_wavefunction_1)
ferro_domain_v = np.load(path+'/'+path_wavefunction_2)
ferro_domain_h = np.load(path+'/'+path_wavefunction_3)
small_polaron  = np.load(path+'/'+path_wavefunction_4)

coupl_object = energy.coupling_of_states(params=params)
diag_object = diag_heff.diagonalization(params=params)
mult_ref_object = diag_heff.multi_ref_ci(params=params)

q = np.array([params['qx'],params['qy']])

print_ref_energies(coupl_object, q, ferro_order, ferro_domain_v, ferro_domain_h, small_polaron)

'''
Create new Reference Ground States from the SCF method of diagonalizing eff. Hamiltonian
'''
iter_number = 5
mult_ref_object.set_phase_bool = False

new_ferro_gs, conv_energ_gs_fo, overlap_arr_fo = mult_ref_object.creat_new_ref_state(iter_number, ferro_order, q)
new_ferro_domain_v, conv_energ_gs_fdv, overlap_arr_fdv = mult_ref_object.creat_new_ref_state(iter_number, ferro_domain_v, q)
new_ferro_domain_h, conv_energ_gs_fdh, overlap_arr_fdh = mult_ref_object.creat_new_ref_state(iter_number, ferro_domain_h, q)
new_small_polaron, conv_energ_gs_sp, overlap_arr_sp = mult_ref_object.creat_new_ref_state(iter_number, small_polaron, q)

print_ref_energies(coupl_object, q, new_ferro_gs, new_ferro_domain_v, new_ferro_domain_h, new_small_polaron)
plot_ref_densities(params, new_ferro_gs, new_ferro_domain_v, new_ferro_domain_h, new_small_polaron)

'''
Diagonalize effective Hamiltonians to get excited states
'''
energy_exc_states, ferro_order_exc_states = diag_object.diag_h_eff(new_ferro_gs)
energy_exc_states, ferro_domain_v_exc_states = diag_object.diag_h_eff(new_ferro_domain_v)
energy_exc_states, ferro_domain_h_exc_states = diag_object.diag_h_eff(new_ferro_domain_h)
energy_exc_states, small_polaron_exc_states = diag_object.diag_h_eff(small_polaron)

'''
Create list of wavefunctions containing GS, Single-, and Double-Excitations
'''
psi_arr = []

psi_arr.append(new_ferro_gs)
psi_arr.append(new_ferro_domain_v)
psi_arr.append(new_ferro_domain_h)
psi_arr.append(new_small_polaron)

psi_arr = mult_ref_object.append_single_excitation(new_ferro_gs, psi_arr, ferro_order_exc_states)
psi_arr = mult_ref_object.append_single_excitation(new_ferro_domain_v, psi_arr, ferro_domain_v_exc_states)
psi_arr = mult_ref_object.append_single_excitation(new_ferro_domain_h, psi_arr, ferro_domain_h_exc_states)
psi_arr = mult_ref_object.append_single_excitation(new_small_polaron, psi_arr, small_polaron_exc_states)

psi_arr = mult_ref_object.append_double_excitations(new_ferro_gs, psi_arr, ferro_order_exc_states)
psi_arr = mult_ref_object.append_double_excitations(new_ferro_domain_v, psi_arr, ferro_domain_v_exc_states)
psi_arr = mult_ref_object.append_double_excitations(new_ferro_domain_h, psi_arr, ferro_domain_h_exc_states)
psi_arr = mult_ref_object.append_double_excitations(new_small_polaron, psi_arr, small_polaron_exc_states)

n_states = len(psi_arr)
q_arr = np.zeros((n_states,2), dtype=complex)

print_number_of_states(n_states, params)

'''
Compute the effective Hamiltonian
'''
h_eff, s_ove = coupl_object.calc_hamiltonian(n_states, psi_arr, q_arr)
print("Finished calculation of Hamiltonian!")