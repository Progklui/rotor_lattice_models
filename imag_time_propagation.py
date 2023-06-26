import numpy as np
import time, os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

''' Program for imaginary time propagation

    ----
    Execution:
        python3 imag_time_propagation.py PATH_TO_INPUT_FILE
    ----

    ----
    Input File Structure: here we give an example input file, except of the # comments it can be used
        {"n": 256,
        "M": 100,
        "Mx": 10,
        "Mx_display": 4, # COMMENT: respective rotor number for which to compute densities and phases
        "My": 10,
        "My_display": 4, # COMMENT: respective rotor number for which to compute densities and phases
        "B": 1.0,
        "tx": 50,
        "ty": 100,
        "V_0": [0.0,6.0,12.0,18.0], # COMMENT: list of potential points
        "qx": 0,
        "qy": 0,
        "init_choice": "uniform", # OPTIONS: uniform, ferro_domain_vertical_wall, ferro_domain_horizontal_wall, small_polaron, external
        "external_wf_tag": "external_ferro-domain", # COMMENT: user defined tag that is added to the file name
        "path_to_input_wavefunction": "matrix_results/psi_rotors_2d_python_M_100_B_1.0_tx_50.0_ty_100.0_Vmin_6.0_Vmax_18.0/\
            psi_rotors_2d_imag_time_prop_M_100_Mx_10_My_10_B_1.0_tx_50.0_ty_100.0_Vmin_6.0_Vmax_18.0_qx_0_qy_0_init_ferro_domain_vertical_wall_V0_18.0.npy", 
        "dt": 0.001,
        "tol": 1e-12}
    ----

    ----
    Description:
        (1) Create objects for handling equations of motion, energy, polaron size
        (2) Create initialization of the wavefunction as specified in the input file
        (3) Imaginary time propagation - evolve until converged
        (4) Store wavefunction, energy, dE_dt and polaron size
        (5) Repeat 3. and 4.
    ----

    ----
    Output:
        (1) stores energies in folder: image_results/psi_rotors_...parameters.../energies/
        (2) stores dE_dt in folder: image_results/psi_rotors_...parameters.../energies/
        (3) stores polaron size in folder: image_results/psi_rotors_...parameters.../polaron_size/
        (4) stores densities in folder: image_results/psi_rotors_...parameters.../rotor_densities/
        (5) stores phases in folder: image_results/psi_rotors_...parameters.../rotor_phases/
        (6) stores wavefunctions in separate files, for every time step - can be analyzed in retrospect
    ----
'''

# import user-defined classes 
import class_energy as energy
import class_equations_of_motion as eom 

import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

import class_visualization as vis
import class_mass_size as mass_size

'''
    MAIN PART:
'''
in_object = h_in.params(on_cluster=False) # object for handling inputs from command line
params = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

V_0_array = np.array(params['V_0'], dtype=float)

# initial wavefunction object
wfn_manip = h_wavef.wavefunc_operations(params=params)
wavefunc_object = h_wavef.wavefunctions(params=params)

orig_init_choice = params['init_choice']
if params['init_choice'] == 'external':
    params['init_choice'] = params['external_wf_tag'] # reasons to create correct tag for storing the results

chosen_My = params['My_display']
chosen_Mx = params['Mx_display']

# energy object
energy_object = energy.energy(params=params) 

# polaron size objects
size_object = mass_size.polaron_size(params=params)
plot_object = vis.configurations(params=params)

# folder structure objects to store results
in_object = h_in.imag_time(params=params)
folder_name_w, file_name_wavefunction = in_object.wavefunction_folder_structure_imag_time_prop(path) 

tic = time.perf_counter() # start timer
eom_object = eom.eom(params=params) 
for V_0 in V_0_array:
    eom_object.V_0 = V_0
    wavefunc_object.V_0 = V_0 

    psi_init = wavefunc_object.create_init_wavefunction(orig_init_choice) # update for small polaron things
    psi_out = eom_object.solve_for_fixed_params_imag_time_prop(psi_init) # psi_out is (My,Mx,n) object

    '''
    save wavefunction
    '''
    np.save(folder_name_w+file_name_wavefunction+str(V_0), psi_out) 

    '''
    compute and save energies
    '''
    energy_object.V_0 = V_0
    E = energy_object.calc_energy(psi_out)
    in_object.save_energies(V_0, E, path)

    '''
    compute and save dE_dt
    '''
    energy_object.V_0 = V_0
    dE_dtx, dE_dty = energy_object.deriv_dE_dt(psi_out)
    in_object.save_dE_dt(V_0, dE_dtx, dE_dty, path)

    '''
    compute, save and plot polaron size
    '''
    sigma = size_object.calc_polaron_size(psi_out, '1')
    in_object.save_polaron_size(V_0, sigma, path)
    plot_object.plot_polaron_size_imag_time(sigma, V_0, path)
    
    '''
    compute and plot rotor densities and phases
    '''
    psi_small = wfn_manip.cut_out_rotor_region(psi_out, chosen_My, chosen_Mx)

    rotor_density = wfn_manip.individual_rotor_density(psi_small, chosen_My, chosen_Mx)
    rotor_phase = wfn_manip.individual_rotor_phase(psi_small, chosen_My, chosen_Mx)

    in_object.save_densities_phases(rotor_density, rotor_phase, V_0, path)
    
    plot_object.plot_single_rotor_density_imag_time(rotor_density, V_0, chosen_My, chosen_Mx, path)
    plot_object.plot_single_rotor_phase_imag_time(rotor_phase, V_0, chosen_My, chosen_Mx, path)

    '''
    delete big objects
    '''
    del sigma
    del psi_out
    gc.collect()

toc = time.perf_counter() # end timer
print("\nExecution time = ", (toc-tic)/60, "min")