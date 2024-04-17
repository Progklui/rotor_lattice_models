import numpy as np
import scipy

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
        ", No. in theory = 1 +", single_exc_number, "+", double_exc_number, " =", (1+single_exc_number+double_exc_number))

def print_ref_energies(coupl_object, q, ref_state):
    e_ref = coupl_object.calc_hamiltonian_matrix_element(ref_state, q, ref_state, q)[0]

    print('\nE (Ferro Order) =', e_ref, '\n')

def plot_ref_densities(params, ref_state):
    polaron_size_object = mass_size.polaron_size(params=params)

    ref_state_sigma = polaron_size_object.calc_polaron_size(ref_state, '1')

    plt.pcolormesh(ref_state_sigma)
    plt.show()

def plot_energies_during_scf(e_ref):
    plt.plot(e_ref.real); plt.show()

in_object = h_in.params(on_cluster=True) # object for handling inputs from command line
print('\nParameters input file:')
params = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

'''
load wavefunctions
'''
path_wavefunction = params['path1']

ref_state = np.load(path+'/'+path_wavefunction)

coupl_object = energy.coupling_of_states(params=params)
diag_object = diag_heff.diagonalization(params=params)
mult_ref_object = diag_heff.multi_ref_ci(params=params)

q = np.array([params['qx'],params['qy']])

print_ref_energies(coupl_object, q, ref_state)
plot_ref_densities(params, ref_state)

'''
Create new Reference Ground States from the SCF method of diagonalizing eff. Hamiltonian
'''
iter_number = 50
mult_ref_object.set_phase_bool = True

new_ref_gs, conv_energ_gs_ref, overlap_arr_ref = mult_ref_object.creat_new_ref_state(iter_number, ref_state, q)

print_ref_energies(coupl_object, q, new_ref_gs)
plot_ref_densities(params, new_ref_gs)
plot_energies_during_scf(conv_energ_gs_ref)

'''
Diagonalize effective Hamiltonians to get excited states
'''
energy_exc_states, ferro_order_exc_states = diag_object.diag_h_eff(new_ref_gs)

'''
Create list of wavefunctions containing GS, Single-, and Double-Excitations
'''
psi_arr = []

psi_arr.append(new_ref_gs)

psi_arr = mult_ref_object.append_single_excitation(new_ref_gs, psi_arr, ferro_order_exc_states)

psi_arr = mult_ref_object.append_double_excitations(new_ref_gs, psi_arr, ferro_order_exc_states)

n_states = len(psi_arr)
q_arr = np.zeros((n_states,2), dtype=complex)

print_number_of_states(n_states, params)

'''
Compute the effective Hamiltonian
'''
h_eff, s_ove = coupl_object.calc_hamiltonian(n_states, psi_arr, q_arr)
print("Finished calculation of Hamiltonian!")

eigen_values, eigen_vector = scipy.linalg.eig(a=h_eff, b=s_ove) # diagonalize effective hamiltonian
order = np.argsort(eigen_values)
eigen_vector = eigen_vector[:,order]
eigen_values = eigen_values[order]

print('min e-val =', np.min(eigen_values))

plt.scatter(np.arange(len(eigen_values)), eigen_values, s=1)
plt.show() 

size_to_show = 400

fig = plt.figure()
pc = plt.pcolormesh(h_eff[0:size_to_show,0:size_to_show][::-1].real)
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=20, length=6)
cbar.set_label(label=r'$\hat{H}_{eff}$', size=20)
plt.show()