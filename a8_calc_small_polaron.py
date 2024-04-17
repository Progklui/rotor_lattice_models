import numpy as np

import time
import os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

''' Computes small polaron wavefunction (i.e. Mathieu functions)

    ----
    Execution:
        python3 a8_calc_small_polaron.py PATH_TO_INPUT_FILE
    ----

    ----
    Input File Structure: here we give an example input file, except of the # comments it can be used
        {"n": 256,
        "M": 100,
        "Mx": 10,
        "My": 10,
        "B": 1.0,
        "tx": 50,
        "ty": 100,
        "V_0": [0.0,6.0,12.0,18.0], # COMMENT: list of potential points
        "qx": 0,
        "qy": 0,
        "init_choice": "small_polaron", # OPTIONS: uniform, ferro_domain_vertical_wall, ferro_domain_horizontal_wall, small_polaron, external
        "external_wf_tag": " ", # COMMENT: user defined tag that is added to the file name
        "path_to_input_wavefunction": " ", 
        "dt": 0.001,
        "tol": 1e-12}
    ----

    ----
    Description:
        (1) Create objects for handling wavefunction, energy, polaron size
        (2) Get analytic small polaron state
        (3) Imaginary time propagation - evolve until converged
        (4) Store wavefunction, energy, dE_dt and polaron size
        (5) Repeat 3. and 4.
    ----

    ----
    Output:
        (1) stores energies in folder: image_results/psi_rotors_...parameters.../energies/
        (2) stores dE_dt in folder: image_results/psi_rotors_...parameters.../energies/
        (3) stores polaron size in folder: image_results/psi_rotors_...parameters.../polaron_size/
        (4) stores wavefunctions in separate files, for every time step - can be analyzed in retrospect
    ----
'''

# import user-defined classes
import class_equations_of_motion as eom 
import class_energy as energy

import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

import class_visualization as vis
import class_mass_size as mass_size

'''
    MAIN PART:
'''
# object for handling inputs from command line
in_object = h_in.params(on_cluster=False) 
params = in_object.get_parameters_imag_time_prop(path+'/', arg=1)

V_0_array = np.array(params['V_0'], dtype=float)

# initial wavefunction object
wavefunc_object = h_wavef.wavefunctions(params=params)

# energy object
energy_object = energy.energy(params=params) 

# polaron size objects
size_object = mass_size.polaron_size(params=params)
plot_object = vis.configurations(params=params)

# folder structure objects to store results
in_object = h_in.imag_time(params=params)
folder_name_w, file_name_wavefunction = in_object.wavefunction_folder_structure_imag_time_prop(path) 
folder_name_p, file_name_size = in_object.polaron_size_results_folder_structure_imag_time_prop(path)

tic = time.perf_counter() # start timer
for V_0 in V_0_array:
    print('Calculating for V_0 =', V_0)

    wavefunc_object.V_0 = V_0
    psi_out = wavefunc_object.create_init_wavefunction('small_polaron')

    '''
    save wavefunction
    '''
    np.save(folder_name_w+file_name_wavefunction+str(V_0), psi_out) 

    '''
    compute and save energies
        TODO: outsource the saving of energies and polaron size here!
    '''
    energy_object.V_0 = V_0
    E = energy_object.calc_energy(psi_out)
    E_analytic = energy_object.analytic_small_polaron_energy()

    E_combined = np.append(E_analytic, E).astype(complex)
    in_object.save_energies(V_0, E_combined, path)

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
    plot_object.plot_polaron_size_imag_time(sigma, V_0, folder_name_p+file_name_size+str(V_0))
    
    del sigma
    del psi_out
    gc.collect()

toc = time.perf_counter() # end timer
print("\nExecution time = ", (toc-tic)/60, "min")