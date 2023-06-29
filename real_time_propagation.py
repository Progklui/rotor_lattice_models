import time, os, sys

path = os.path.dirname(__file__) 
sys.path.append(path)

''' Program for real time propagation, i.e. time quench of an initially uniformly oriented rotor grid

    ----
    Execution:
        python3 real_time_propagation.py PATH_TO_INPUT_FILE
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
        "V_0": 80.0, # COMMENT: just one potential point here
        "qx": 0,
        "qy": 0,
        "init_choice": "uniform", # OPTIONS: uniform (always here!), ferro_domain_vertical_wall, ferro_domain_horizontal_wall, small_polaron, external
        "external_wf_tag": " ", # COMMENT: user defined tag that is added to the file name
        "path_to_input_wavefunction": " ", 
        "dt": 0.001,
        "tol": 1e-12}
    ----

    ----
    Description:
        (1) Create objects for computing green function and energy
        (2) Create UNIFORM initialization of the wavefunction
        (3) Real time propagate wavefunction for dt
        (4) Compute overlap (i.e. Green func here), norm sum of every single rotor and energy
        (5) Update psi_curr and evolve psi_curr again for dt
        (6) Repeat 3. to 5.
    ----

    ----
    Output:
        (1) stores energies and green func overlaps in folder: image_results/psi_rotors_...parameters.../green_functions/
        (2) stores dE_dt in folder: image_results/psi_rotors_...parameters.../energies/
        (3) stores densities in folder: image_results/psi_rotors_...parameters.../rotor_densities/
        (3) stores phases in folder: image_results/psi_rotors_...parameters.../rotor_phases/
        (3) stores polaron size in folder: image_results/psi_rotors_...parameters.../polaron_size/
        (4) stores wavefunctions in separate files, for every time step - can be analyzed in retrospect
    ----
'''

# import user-defined classes 
import class_equations_of_motion as eom 
import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

'''
    MAIN PART:
'''
in_object = h_in.params(on_cluster=True) # object for handling inputs from command line
params = in_object.get_parameters_real_time_prop(path+'/', arg=1)

# create uniform initial wavefunction
wavefunc_object = h_wavef.wavefunctions(params=params)
psi_init = wavefunc_object.create_init_wavefunction('uniform') 

tic = time.perf_counter() # start timer

eom_object = eom.eom(params=params) # equations of motion object
eom_object.V_0 = float(params['V_0'])
eom_object.solve_for_fixed_params_real_time_prop(psi_init, path) # all objects for the calculation are calculated there!

toc = time.perf_counter() # end timer
print("\nExecution time = ", (toc-tic)/60, "min")