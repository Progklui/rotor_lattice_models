import numpy as np
import time, os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

'''
    - Goal of the code: time quench of an initially uniformly oriented rotor grid
    - Call: python3 real_time_propagation.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - Uniform initialization of the rotor grid
        - Propagate for dt
        - Compute properties like Overlap (=Green f. here), Norm and Energy

    - Main output: (a) stores energies in folder: image_results/psi_rotors_...parameters.../energies/
                   (b) stores polaron size in folder: image_results/psi_rotors_...parameters.../polaron_size/
                   (c) stores wavefunctions in separate files, for every time step - can be analyzed in retrospect

    - Logic: 
        1. Create diverse objects for handling equations of motion, energy, polaron size
        2. Create uniform initialization of the wavefunction
        3. Imaginary time propagation - evolve until converged
        4. Store wavefunction, energy and polaron size
        5. Repeat 3. and 4.
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

# create initial wavefunction
wavefunc_object = h_wavef.wavefunctions(params=params)
psi_init = wavefunc_object.create_init_wavefunction(params['init_choice']) 

if params['init_choice'] == 'external':
    params['init_choice'] = params['external_wf_tag'] # reasons to create correct tag for storing the results

# energy object
energy_object = energy.energy(params=params) 

# polaron size objects
size_object = mass_size.polaron_size(params=params)
plot_object = vis.configurations(params=params)

# folder structure objects to store results
in_object = h_in.imag_time(params=params)
folder_name_w, file_name_wavefunction = in_object.wavefunction_folder_structure_imag_time_prop(path) 

folder_name_e, file_name_energies = in_object.energy_results_folder_structure_imag_time_prop(path) 
folder_name_de_dt, file_name_de_dt = in_object.t_deriv_energy_results_folder_structure_imag_time_prop(path)
folder_name_p, file_name_size = in_object.polaron_size_results_folder_structure_imag_time_prop(path)

tic = time.perf_counter() # start timer
eom_object = eom.eom(params=params) 
for V_0 in V_0_array:
    eom_object.V_0 = V_0
    wavefunc_object.V_0 = V_0 

    psi_init = wavefunc_object.create_init_wavefunction(params['init_choice']) # update for small polaron things
    psi_out = eom_object.solve_for_fixed_params_imag_time_prop(psi_init) 

    # save wavefunction
    np.save(folder_name_w+file_name_wavefunction+str(V_0), psi_out) 

    # save energies
    energy_object.V_0 = V_0
    E = energy_object.calc_energy(psi_out)
    with open(folder_name_e+file_name_energies, 'a') as energy_file:
        write_string = str(V_0)+' '+str(E[0])+' '+str(E[1])+' '+str(E[2])+' '+str(E[3])+'\n'
        energy_file.write(write_string)

    # save dE_dt
    energy_object.V_0 = V_0
    dE_dtx, dE_dty = energy_object.deriv_dE_dt(psi_out)
    with open(folder_name_de_dt+file_name_de_dt, 'a') as de_dt_file:
        write_string = str(V_0)+' '+str(dE_dtx)+' '+str(dE_dty)+'\n'
        de_dt_file.write(write_string)

    # save and plot polaron size
    sigma = size_object.calc_polaron_size(psi_out, '1')
    np.savetxt(folder_name_p+file_name_size+str(V_0)+'.out', (sigma))
    plot_object.plot_polaron_size_imag_time(sigma, V_0, folder_name_p+file_name_size+str(V_0))
    
    del sigma
    del psi_out
    gc.collect()

toc = time.perf_counter() # end timer
print("\nExecution time = ", (toc-tic)/60, "min")