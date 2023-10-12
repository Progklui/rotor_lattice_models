import numpy as np
import matplotlib.pyplot as plt
import time, os, sys, gc

import h5py 

from scipy.integrate import solve_ivp

import class_energy as energy
import class_equations_of_motion as eom 

import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

import class_visualization as vis
import class_mass_size as mass_size

def get_file_name(params):
    ''' 
    ----
    Description: get file name for a run
    ----

    ----
    Inputs:
        params (data dict): defines all the calculation variables in a dictionary
    ----

    ----
    Outputs:
        file_name (string): file name
    ----
    '''
    Mx = params["Mx"]
    My = params["My"] 
    tx = params["tx"]
    ty = params["ty"]
    V_0 = params["V_0"]
    B = params["B"]

    file_name = 'tx_'+str(tx)+'_ty_'+str(ty)+'_V0_'+str(V_0)+'_B_'+str(B)+'_Mx_'+str(Mx)+'_My_'+str(My)+'.hfd5'
    return file_name

def save_calculation_run(psi, E_imag, epsilon_imag, params, folder):
    ''' 
    ----
    Description: stores the wavefunction, energies and epsilon criterion during imag time propagation, hdf file format
    ----

    ----
    Inputs:
        psi (3-dimensional: (My,Mx,n)): wavefunction to store
        E_imag (1-dimensional: (number of runs)): energies during the imag time propagation
        epsilon_imag (1-dimensional: (number of runs)): convergence criterion
        params (data dict): defines all the calculation variables in a dictionary
        folder (string): folder path in form '../../'
    ----

    ----
    Outputs:
        file_name (string): file name
    ----
    '''

    file_name = get_file_name(params)
    with h5py.File(folder+file_name, "a") as f:
        if 'phi' in f:
            phi_data = f['phi']
            phi_data[...] = psi
            phi_data.attrs['Mx'] = params['Mx']
            phi_data.attrs['My'] = params['My']
            phi_data.attrs['n']  = params['n']
            phi_data.attrs['tx'] = params['tx']
            phi_data.attrs['ty'] = params['ty']
            phi_data.attrs['V_0'] = params['V_0']
            phi_data.attrs['B'] = params['B']
            phi_data.attrs['qx'] = params['qx']
            phi_data.attrs['qy'] = params['qy']
            phi_data.attrs['tol'] = params['tol']
            phi_data.attrs['dt'] = params['dt']
            phi_data.attrs['init_choice'] = params['init_choice']
        else:
            phi_data = f.create_dataset('phi', data=psi)
            phi_data.attrs['Mx'] = params['Mx']
            phi_data.attrs['My'] = params['My']
            phi_data.attrs['n']  = params['n']
            phi_data.attrs['tx'] = params['tx']
            phi_data.attrs['ty'] = params['ty']
            phi_data.attrs['V_0'] = params['V_0']
            phi_data.attrs['B'] = params['B']
            phi_data.attrs['qx'] = params['qx']
            phi_data.attrs['qy'] = params['qy']
            phi_data.attrs['tol'] = params['tol']
            phi_data.attrs['dt'] = params['dt']
            phi_data.attrs['init_choice'] = params['init_choice']

        if 'e_imag_time_prop' in f:
            e_evo_data = f['e_imag_time_prop']
            e_evo_data[...] = np.array(E_imag)
        else:
            f.create_dataset('e_imag_time_prop', data=np.array(E_imag))

        if 'epsilon_imag_prop' in f:
            epsilon_data = f['epsilon_imag_prop']
            epsilon_data[...] = epsilon_imag
        else:
            f.create_dataset('epsilon_imag_prop', data=epsilon_imag)

    return

def get_psi(file_path):
    with h5py.File(file_path, 'r') as f:
        phi = f['phi']
        e_evo = f['e_imag_time_prop']
        epsilon_evo = f['epsilon_imag_prop']

        params = {"n": int(phi.attrs['n']),
                "Mx": int(phi.attrs['Mx']),
                "Mx_display": 4,
                "converge_new_lattice": "no",
                "My": int(phi.attrs['My']),
                "M": int(phi.attrs['Mx']*phi.attrs['My']),
                "My_display": 4,
                "B": float(phi.attrs['B']),
                "tx": float(phi.attrs['tx']),
                "ty": float(phi.attrs['ty']),
                "V_0": float(phi.attrs['V_0']),
                "qx": int(phi.attrs['qx']),
                "qy": int(phi.attrs['qy']),
                "init_choice": "ferro_domain_vertical_wall",
                "external_wf_tag": " ",
                "excitation_no": 0,
                "angle_pattern": [0,0,0,0],
                "V_0_pattern": [0,0,0,0],
                "n_states": 0,
                "path_to_input_wavefunction": " ",
                "dt": float(phi.attrs['dt']),
                "tol": float(phi.attrs['tol'])}

        phi = f['phi'][...]

    return phi, params

def solve_imag_time_prop(params):
    Mx = params["Mx"]
    My = params["My"]

    n = params["n"]
    V_0 = params["V_0"]

    '''
    EOM and wavefunction manip objects
    '''
    wfn_manip = h_wavef.wavefunc_operations(params=params)
    wavefunc_object = h_wavef.wavefunctions(params=params)
    eom_object = eom.eom(params=params) 

    eom_object.V_0 = V_0
    wavefunc_object.V_0 = V_0 

    psi_init = wavefunc_object.create_init_wavefunction(params['init_choice']) # update for small polaron things
    psi_init = wfn_manip.reshape_one_dim(psi_init)
    
    '''
    Energy Objects
    '''
    energy_object = energy.energy(params=params)
    overlap_object = energy.coupling_of_states(params=params) # needed for overlap calculations
        
    energy_object.V_0 = V_0
    overlap_object.V_0 = V_0

    ''' 
    Lambda expression of right-hand-side of e.o.m
    '''
    func = eom_object.create_integration_function_imag_time_prop() 

    iter = 0
    epsilon = 1 
    tol = params['tol']
    dt  = params['dt']

    E_converge_list = []
    epsilon_list = []
    
    while epsilon > tol:
        '''
        imag time evolution for dt
        '''
        sol = solve_ivp(func, [0,dt], psi_init, method='RK45', rtol=1e-9, atol=1e-9) # method='RK45','DOP853'

        '''
        normalize
        '''
        psi_iter = sol.y.T[-1]
        psi_iter = wfn_manip.normalize_wf(psi_iter, shape=(int(Mx*My),n))

        '''
        compute and save energy and epsilon criterion
        '''
        E = energy_object.calc_energy(psi_iter)
        E_converge_list.append(E[0].real)

        epsilon = eom_object.epsilon_criterion_single_rotor(psi_iter, psi_init)
        epsilon_list.append(epsilon)
        #print('V_0 =', V_0, ', iter step = ' + str(iter+1)+", E =", E[0].real, ", epsilon =", epsilon, "\n")

        '''
        update psi_init
        '''
        psi_init = wfn_manip.reshape_one_dim(psi_iter)

        iter = iter + 1

    psi_out = wfn_manip.reshape_three_dim(psi_init)

    return psi_out, np.array(E_converge_list), np.array(epsilon_list)

def get_tx_ty(sum, diff):
    '''  
    Return tx and ty tunneling rates
    '''
    tx = 0.5*(sum+diff)
    ty = 0.5*(sum-diff)
    return tx, ty

params = {"n": 256,
"M": 64,
"Mx": 8,
"Mx_display": 4,
"converge_new_lattice": "no",
"My": 8,
"My_display": 4,
"B": 1.0,
"tx": 100,
"ty": 100,
"V_0": 150.0,
"qx": 0,
"qy": 0,
"init_choice": "ferro_domain_vertical_wall",
"external_wf_tag": " ",
"excitation_no": 11,
"angle_pattern": [0,0,0,0],
"V_0_pattern": [150,150,150,150],
"n_states": 0,
"path_to_input_wavefunction": " ",
"dt": 0.001,
"tol": 1e-9}

x = (2*np.pi/params["n"])*np.arange(params["n"])

Mx = params["Mx"]
My = params["My"]

n = params["n"]

tx = params["tx"]
ty = params["ty"]

B = params["B"]
V_0 = params["V_0"]

scale = B 

exc_number = params["excitation_no"]

sum_tx_ty = 200
diff_tx_ty = np.array([-200,-175,-150,-125,-100,-75,-50,-25,-1,1,25,50,75,100,125,150,175,200])
V_0_arr = np.arange(0,355,5) #np.array([0,25,50,75,100,125,150,175,200,225,250])

print(V_0_arr)
for i in diff_tx_ty:
    print(get_tx_ty(sum_tx_ty,i))


params["init_choice"] = "ferro_domain_horizontal_wall"
params["Mx"] = 16
params["My"] = 4
for j in range(len(diff_tx_ty)):
    tx, ty = get_tx_ty(sum_tx_ty, diff_tx_ty[j])

    params["tx"] = tx
    params["ty"] = ty
    
    V_0_arr_run = V_0_arr # make_fine_grained_potential(float(tx),float(ty),V_0_arr)
    for i in range(len(V_0_arr_run)):
        params["V_0"] = V_0_arr_run[i]
        
        '''  
        Imag Time Prop
        '''
        psi, E_evo, epsilon_evo = solve_imag_time_prop(params)
        print('V_0 = ', V_0_arr[i], ', tx-ty =', diff_tx_ty[j], ', E =', E_evo[-1])

        '''  
        Store Results
        '''
        folder = 'results/phase_diagram/fdh/'
        save_calculation_run(psi, E_evo, epsilon_evo, params, folder)


params["init_choice"] = "small_polaron"
params["Mx"] = 8
params["My"] = 8
for j in range(len(diff_tx_ty)):
    tx, ty = get_tx_ty(sum_tx_ty, diff_tx_ty[j])

    params["tx"] = tx
    params["ty"] = ty
    
    V_0_arr_run = V_0_arr # make_fine_grained_potential(float(tx),float(ty),V_0_arr)
    for i in range(len(V_0_arr_run)):
        params["V_0"] = V_0_arr_run[i]
        
        '''  
        Imag Time Prop
        '''
        psi, E_evo, epsilon_evo = solve_imag_time_prop(params)
        print('V_0 = ', V_0_arr[i], ', tx-ty =', diff_tx_ty[j], ', E =', E_evo[-1])

        '''  
        Store Results
        '''
        folder = 'results/phase_diagram/sp/'
        save_calculation_run(psi, E_evo, epsilon_evo, params, folder)