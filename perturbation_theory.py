import numpy as np

import time, os, sys, gc
path = os.path.dirname(__file__) 

import h5py 



import class_energy as energy
import class_equations_of_motion as eom 

import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

def quick_pot_scan(params, V_0_arr, folder):
    h5_io_object = h_in.io_hdf5()

    print('\nvGH Calculations:')
    E_arr = np.zeros(len(V_0_arr), dtype=complex)
    for i in range(len(V_0_arr)):
        print('V0 =', V_0_arr[i])
        params["V_0"] = V_0_arr[i]

        eom_object = eom.eom(params=params)
        wavefunc_object = h_wavef.wavefunctions(params=params)
        wfn_manip = h_wavef.wavefunc_operations(params=params)
        ''' 
        Init wavefunction
        '''
        psi_init = wavefunc_object.create_init_wavefunction(params['init_choice'])
        psi_init = wfn_manip.reshape_one_dim(psi_init)
        ''' 
        Imaginary Time Propagation
        '''
        psi, E_evo, epsilon_evo = eom_object.solve_for_fixed_params_imag_time_prop_new(psi_init)
        E_arr[i] = E_evo[-1]

        h5_io_object.save_calculation_run(psi, E_evo, epsilon_evo, params, folder)

    return E_arr

def get_Ep_from_E_vGH(E_vGH, tx, ty):
    return E_vGH+2*tx+2*ty

def get_E_from_Ep(E_vGH, tx, ty):
    return E_vGH-2*tx-2*ty

def get_E_arr(V_0_range, folder, check_sys, check_sym):
    h5_io_object = h_in.io_hdf5()
    
    E_arr = np.zeros(len(V_0_range), dtype=complex)
    for i in range(len(V_0_range)):
        check_sys_n = check_sys+'_V0_'+str(V_0_range[i])
        file_list = [f for f in os.listdir(folder) if check_sys_n in f and check_sym in f]
        
        if len(file_list) > 1: 
            break
        for file in file_list:
            file_name = folder+file
            psi, params = h5_io_object.get_psi(file_name)

            energy_object = energy.energy(params=params)
            E, E_T, E_B, E_V = energy_object.calc_energy(psi)
            E_arr[i] = E
    return E_arr

params = {"n": 256,
"M": 36,
"Mx": 2,
"Mx_display": 4,
"converge_new_lattice": "no",
"My": 256,
"My_display": 4,
"B": 1.0,
"tx": 100,
"ty": 100,
"V_0": 0.0,
"qx": 0,
"qy": 0,
"init_choice": "uniform",
"external_wf_tag": " ",
"excitation_no": 0,
"angle_pattern": [0,0,0,0],
"V_0_pattern": [0,0,0,0],
"n_states": 0,
"path_to_input_wavefunction": " ",
"dt": 0.001,
"tol": 1e-10}

x = (2*np.pi/params["n"])*np.arange(params["n"])

''' 
I/O Object
'''
h5_io_object = h_in.io_hdf5()

try:
    calc_flag = str(sys.argv[1])
    B = float(sys.argv[2])
    params["B"] = B
    print("\nInitialization:", calc_flag)
    print("B =", params["B"], '\n')
except:
    print(" "); print("Please provide which initialization phase should be chosen!"); print(' ')
    pass

V_0_range = np.linspace(0,10,21)

params["B"] = B
params["init_choice"] = calc_flag

params["tx"] = 0
params["ty"] = 100
folder = path+'/results/perturbation_theory/tx_0_ty_100/'
E_vGH1 = quick_pot_scan(params, V_0_range, folder)

#params["tx"] = 50
#params["ty"] = 150
#folder = path+'/results/perturbation_theory/tx_50_ty_150/'
#E_vGH2 = quick_pot_scan(params, V_0_range, folder)
#
#params["tx"] = 10
#params["ty"] = 190
#folder = path+'/results/perturbation_theory/tx_10_ty_190/'
#E_vGH3 = quick_pot_scan(params, V_0_range, folder)
