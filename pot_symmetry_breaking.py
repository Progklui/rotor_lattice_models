import numpy as np
import matplotlib.pyplot as plt
import time, os, sys, gc

import h5py 

from scipy.integrate import solve_ivp

import class_energy as energy
import class_equations_of_motion as eom 

import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

path = os.path.dirname(__file__) 

def make_scan(delta_angle, sum_tx_ty, V_0_arr, params, folder):
    for j in range(len(delta_angle)):
        params["tx"] = 0.5*sum_tx_ty
        params["ty"] = 0.5*sum_tx_ty

        params["angle_pattern"] = [-delta_angle[j],delta_angle[j],delta_angle[j],-delta_angle[j]] 

        V_0_arr_run = V_0_arr # make_fine_grained_potential(float(tx),float(ty),V_0_arr)
        for i in range(len(V_0_arr_run)):
            params["V_0"] = V_0_arr_run[i]

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
            psi, E_evo, epsilon_evo = eom_object.solve_for_fixed_params_imag_time_prop_sym_breaking_int_new(psi_init)
            print('V_0 = ', V_0_arr[i], ', delta phi =', delta_angle[j], ', E =', E_evo[-1])

            '''  
            Store Results
            '''
            h5_io_object.save_calculation_run_sym_breaking(psi, E_evo, epsilon_evo, params, folder)
    
    return

def get_tx_ty(sum, diff):
    '''  
    Return tx and ty tunneling rates
    '''
    tx = 0.5*(sum+diff)
    ty = 0.5*(sum-diff)
    return tx, ty

params = {"n": 256,
"M": 0,
"Mx": 0,
"Mx_display": 4,
"converge_new_lattice": "no",
"My": 0,
"My_display": 4,
"B": 1.0,
"tx": 0,
"ty": 0,
"V_0": 0.0,
"qx": 0,
"qy": 0,
"init_choice": "ferro_domain_vertical_wall",
"external_wf_tag": " ",
"excitation_no": 0,
"angle_pattern": [0,0,0,0],
"V_0_pattern": [150,150,150,150],
"n_states": 0,
"path_to_input_wavefunction": " ",
"dt": 0.001,
"tol": 1e-10}

x = (2*np.pi/params["n"])*np.arange(params["n"])

''' 
1. I/O Object: get flag for which phase to compute the phase diagram
'''
h5_io_object = h_in.io_hdf5()

try:
    calc_flag = sys.argv[1]
    B = float(sys.argv[2])
    params["B"] = B
    print("\nInitialization:", calc_flag)
    print("B =", params["B"], '\n')
except:
    print(" "); print("Please provide which initialization phase should be chosen!"); print(' ')
    pass

''' 
2. Important: here define the symmetries and potential points
'''
sum_tx_ty = 200
delta_angle = np.array([-0.2,-0.1,0,0.1,0.2])
diff_tx_ty = np.array([0]) #175, 200 #np.array([0]) # np.array([-200,-175,-150,-125,-100,-75,-50,-25,-1,1,25,50,75,100,125,150,175,200])
V_0_arr = np.arange(10,310,10) #np.arange(70,355,5) #np.array([315,320,325,330,335,340,345,350]) #np.arange(0,355,5)

print('V_0 =', V_0_arr, '\n')
for i in diff_tx_ty:
    print('tx,ty =', get_tx_ty(sum_tx_ty,i))
print(' ')

'''
3. Calculate energies for the different regimes
'''
if calc_flag == 'fo':
    params["init_choice"] = "uniform"
    params["Mx"] = 16
    params["My"] = 16
    folder = path+'/results/phase_diagram_sym_breaking_angle/fo/'
    make_scan(delta_angle, sum_tx_ty, V_0_arr, params, folder)
elif calc_flag == 'fdv':
    params["init_choice"] = "ferro_domain_vertical_wall"
    params["Mx"] = 4
    params["My"] = 32 #256
    folder = path+'/results/phase_diagram_sym_breaking_angle/fdv/'
    make_scan(delta_angle, sum_tx_ty, V_0_arr, params, folder)
elif calc_flag == 'fdh':
    params["init_choice"] = "ferro_domain_horizontal_wall"
    params["Mx"] = 32
    params["My"] = 4
    folder = path+'/results/phase_diagram_sym_breaking_angle/fdh/'
    make_scan(delta_angle, sum_tx_ty, V_0_arr, params, folder)
elif calc_flag == 'sp':
    params["init_choice"] = "small_polaron"
    params["Mx"] = 8
    params["My"] = 8
    folder = path+'/results/phase_diagram_sym_breaking_angle/sp/'
    make_scan(delta_angle, sum_tx_ty, V_0_arr, params, folder)