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
        ", No. in theory = 4*(1 +", single_exc_number, "+", double_exc_number, ") =", 4*(1+single_exc_number+double_exc_number))

def print_ref_energies(coupl_object, q, ferro_order, ferro_domain_v, ferro_domain_h, small_polaron):
    e_order = coupl_object.calc_hamiltonian_matrix_element(ferro_order, q, ferro_order, q)[0]
    e_d_v = coupl_object.calc_hamiltonian_matrix_element(ferro_domain_v, q, ferro_domain_v, q)[0]
    e_d_h = coupl_object.calc_hamiltonian_matrix_element(ferro_domain_h, q, ferro_domain_h, q)[0]
    e_s = coupl_object.calc_hamiltonian_matrix_element(small_polaron, q, small_polaron, q)[0]

    print('\nE (Ferro Order)        =', e_order)
    print('E (Ferro Domain Vert.) =', e_d_v)
    print('E (Ferro Domain Hor.)  =', e_d_h)
    print('E (Small Polaron)      =', e_s, '\n')

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

def plot_energies_during_scf(e_fo, e_fdv, e_fdh, e_sp, scale, params):
    A = 6
    plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
    font_size = 20

    fig = plt.figure()
    gs = fig.add_gridspec(4, hspace=0.3)
    axs = gs.subplots(sharex=True)

    axs[0].plot(e_fo.real/scale, label=r'Ferro Order')
    axs[1].plot(e_fdv.real/scale, label=r'Ferro-Domain Vertical')
    axs[2].plot(e_fdh.real/scale, label=r'Ferro-Domain Horizontal')
    axs[3].plot(e_sp.real/scale, label=r'Small Polaron') 

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    plt.xlabel(r'SCF Iteration Step $i$') #, fontsize=font_size)
    #axs[0].set_ylabel(r'$E/B$') #, fontsize=font_size)
    #axs[1].set_ylabel(r'$E/B$')
    #axs[2].set_ylabel(r'$E/B$')
    #axs[3].set_ylabel(r'$E/B$')

    fig.text(0.025, 0.5, r'$E_i/B$', va='center', rotation='vertical')

    axs[0].tick_params(axis='x', direction='in', length=5, top=True, bottom=False)
    axs[1].tick_params(axis='x', direction='in', length=5, top=False, bottom=False)
    axs[2].tick_params(axis='x', direction='in', length=5, top=False, bottom=False)
    axs[3].tick_params(axis='x', direction='in', length=5, top=False, bottom=True)

    axs[0].tick_params(axis='y', direction='in', length=5, right=True)
    axs[1].tick_params(axis='y', direction='in', length=5, right=True)
    axs[2].tick_params(axis='y', direction='in', length=5, right=True)
    axs[3].tick_params(axis='y', direction='in', length=5, right=True)

    #axs[0].tick_params(which='minor', axis='y', direction='in', right=True)

    try: os.makedirs(path+'/'+params["results_path"])
    except FileExistsError: pass
    plt.savefig(path+'/'+params["results_path"]+'energies_during_scf_evolution.svg', dpi=400)
    plt.close()

def plot_overlap_during_scf(e_fo, e_fdv, e_fdh, e_sp, params):
    A = 6
    plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
    font_size = 20

    fig = plt.figure()
    gs = fig.add_gridspec(4, hspace=0.3)
    axs = gs.subplots(sharex=True)

    axs[0].plot(e_fo.real, label=r'Ferro Order')
    axs[1].plot(e_fdv.real, label=r'Ferro-Domain Vertical')
    axs[2].plot(e_fdh.real, label=r'Ferro-Domain Horizontal')
    axs[3].plot(e_sp.real, label=r'Small Polaron') 

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    plt.xlabel(r'SCF Iteration Step $i$') #, fontsize=font_size)
    #axs[0].set_ylabel(r'Overlap') #, fontsize=font_size)
    #axs[1].set_ylabel(r'Overlap')
    #axs[2].set_ylabel(r'Overlap')
    #axs[3].set_ylabel(r'Overlap')

    fig.text(0.025, 0.5, r'Overlap $\langle\psi_{i-1}|\psi_{i}\rangle$', va='center', rotation='vertical')

    axs[0].tick_params(axis='x', direction='in', length=5, top=True, bottom=False)
    axs[1].tick_params(axis='x', direction='in', length=5, top=False, bottom=False)
    axs[2].tick_params(axis='x', direction='in', length=5, top=False, bottom=False)
    axs[3].tick_params(axis='x', direction='in', length=5, top=False, bottom=True)

    axs[0].tick_params(axis='y', direction='in', length=5, right=True)
    axs[1].tick_params(axis='y', direction='in', length=5, right=True)
    axs[2].tick_params(axis='y', direction='in', length=5, right=True)
    axs[3].tick_params(axis='y', direction='in', length=5, right=True)

    try: os.makedirs(path+'/'+params["results_path"])
    except FileExistsError: pass
    plt.savefig(path+'/'+params["results_path"]+'overlap_during_scf_evolution.svg', dpi=400)
    plt.close()

def plot_e_vals(params, e_vals, scale):
    A = 6
    plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
    font_size = 10

    plt.scatter(np.arange(len(e_vals)), e_vals/scale, s=1)
    
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.xlabel(r'Number of Eigenvalue $n$', fontsize=font_size)
    plt.ylabel(r'$E_n/B$', fontsize=font_size)

    plt.tick_params(axis='x', direction='in', length=8, top=True)
    plt.tick_params(axis='y', direction='in', length=8, right=True)
    plt.tick_params(which='minor', axis='y', direction='in', right=True)

    try: os.makedirs(path+'/'+params["results_path"])
    except FileExistsError: pass
    plt.savefig(path+'/'+params["results_path"]+'eigen_values_'+str(params["excitation_no"])+'.svg', dpi=400)
    plt.close()


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
#plot_ref_densities(params, ferro_order, ferro_domain_v, ferro_domain_h, small_polaron)

'''
Create new Reference Ground States from the SCF method of diagonalizing eff. Hamiltonian
'''
iter_number = params["iter_number"]
mult_ref_object.set_phase_bool = bool(params["control_phase"])

new_ferro_gs, conv_energ_gs_fo, overlap_arr_fo = mult_ref_object.creat_new_ref_state(iter_number, ferro_order, q)
new_ferro_domain_v, conv_energ_gs_fdv, overlap_arr_fdv = mult_ref_object.creat_new_ref_state(iter_number, ferro_domain_v, q)
new_ferro_domain_h, conv_energ_gs_fdh, overlap_arr_fdh = mult_ref_object.creat_new_ref_state(iter_number, ferro_domain_h, q)
new_small_polaron, conv_energ_gs_sp, overlap_arr_sp = mult_ref_object.creat_new_ref_state(iter_number, small_polaron, q)

print_ref_energies(coupl_object, q, new_ferro_gs, new_ferro_domain_v, new_ferro_domain_h, new_small_polaron)
#plot_ref_densities(params, new_ferro_gs, new_ferro_domain_v, new_ferro_domain_h, new_small_polaron)
plot_energies_during_scf(conv_energ_gs_fo, conv_energ_gs_fdv, conv_energ_gs_fdh, conv_energ_gs_sp, params["B"], params)
plot_overlap_during_scf(overlap_arr_fo, overlap_arr_fdv, overlap_arr_fdh, overlap_arr_sp, params)

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

if params["couple_state1"] == "FO":
    psi_arr.append(new_ferro_gs)
if params["couple_state2"] == "FDv":
    psi_arr.append(new_ferro_domain_v)
if params["couple_state3"] == "FDh":
    psi_arr.append(new_ferro_domain_h)
if params["couple_state4"] == "SP":
    psi_arr.append(new_small_polaron)

if params["couple_state1"] == "FO":
    psi_arr = mult_ref_object.append_single_excitation(new_ferro_gs, psi_arr, ferro_order_exc_states)
if params["couple_state2"] == "FDv":
    psi_arr = mult_ref_object.append_single_excitation(new_ferro_domain_v, psi_arr, ferro_domain_v_exc_states)
if params["couple_state3"] == "FDh":
    psi_arr = mult_ref_object.append_single_excitation(new_ferro_domain_h, psi_arr, ferro_domain_h_exc_states)
if params["couple_state4"] == "SP":
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

print('\nH_eff min =', np.min(h_eff))
print('H_eff max =', np.max(h_eff))

eigen_values, eigen_vector = scipy.linalg.eig(a=h_eff, b=s_ove) # diagonalize effective hamiltonian
order = np.argsort(eigen_values)
eigen_vector = eigen_vector[:,order]
eigen_values = eigen_values[order]

print('\nMin e-val =', np.min(eigen_values))
print('Max e-val =', np.max(eigen_values))

plot_e_vals(params, eigen_values, params["B"])

with open(path+'/'+params["results_path"]+'e_values_exc_no_'+str(params["excitation_no"])+'.out', 'a') as e_val_file:
    write_string = ''
    for i in range(n_states):
        write_string += str(i)+' '+str(eigen_values[i])+'\n'
    e_val_file.write(write_string)

np.save(path+'/'+params["results_path"]+'h_eff_exc_no_'+str(params["excitation_no"]), (h_eff))
np.save(path+'/'+params["results_path"]+'s_ove_exc_no_'+str(params["excitation_no"]), (s_ove))

size_to_show = 400

fig = plt.figure()
pc = plt.pcolormesh(h_eff.real, cmap='RdBu')
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=20, length=6)
cbar.set_label(label=r'$\hat{H}_{eff}$', size=20)
plt.show()