import numpy as np
import class_equations_of_motion as eom 

import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

def make_size_scan(Mx_list, My_list, params, folder):
    h5_io_object = h_in.io_hdf5()

    for i in range(len(Mx_list)):
        params["My"] = My_list[i]
        params["Mx"] = Mx_list[i]
        print('(My,Mx) =', params["My"], params["Mx"])

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
        E_evo[i] = E_evo[-1]

        '''  
        Store Results
        '''
        h5_io_object.save_calculation_run(psi, E_evo, epsilon_evo, params, folder)
    return 

params = {"n": 256,
"M": 36,
"Mx": 6,
"Mx_display": 4,
"converge_new_lattice": "no",
"My": 6,
"My_display": 4,
"B": 1.0,
"tx": 100,
"ty": 100,
"V_0": 150.0,
"qx": 0,
"qy": 0,
"init_choice": "uniform",
"external_wf_tag": " ",
"excitation_no": 11,
"angle_pattern": [0,0,0,0],
"V_0_pattern": [0,0,0,0],
"n_states": 0,
"path_to_input_wavefunction": " ",
"dt": 0.001,
"tol": 1e-7}

x = (2*np.pi/params["n"])*np.arange(params["n"])

''' 
I/O Object
'''
h5_io_object = h_in.io_hdf5()


'''
1. Scan for Ferro-Order
'''
Mx_list = 2**(np.arange(2,7))
My_list = 2**(np.arange(2,7))

params["init_choice"] = "uniform"
folder = 'results/numerics_verification/fo/'

make_size_scan(Mx_list, My_list, params, folder)


'''
2. Scan for vertical Ferro-Domain
'''
My_list = 2**(np.arange(2,9))
Mx_list = np.ones(len(My_list), dtype=int)

params["init_choice"] = "ferro_domain_vertical_wall"
folder = 'results/numerics_verification/fdv/'

make_size_scan(Mx_list, My_list, params, folder)


'''
3. Scan for horizontal Ferro-Domain
'''
Mx_list = 2**(np.arange(2,9)) 
My_list = np.ones(len(Mx_list), dtype=int)

params["init_choice"] = "ferro_domain_horizontal_wall"
folder = 'results/numerics_verification/fdh/'

make_size_scan(Mx_list, My_list, params, folder)


'''
4. Scan for Small Polaron
'''
Mx_list = 2**(np.arange(2,7))
My_list = 2**(np.arange(2,7))

params["init_choice"] = "small_polaron"
folder = 'results/numerics_verification/sp/'

make_size_scan(Mx_list, My_list, params, folder)