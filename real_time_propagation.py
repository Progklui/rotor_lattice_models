import numpy as np
import time, os, sys

path = os.path.dirname(__file__) 
sys.path.append(path)

'''
    - Goal of the code: time quench of an initially uniformly oriented rotor grid
    - Call: python3 real_time_propagation.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - Uniform initialization of the rotor grid
        - Propagate for dt
        - Compute properties like Overlap (=Green f. here), Norm and Energy

    - Main output: (a) stores energies in folder: image_results/psi_rotors_...parameters.../green_functions/
                   (b) stores wavefunctions in separate files, for every time step - can be analyzed in retrospect
'''

# import user-defined classes 
import class_equations_of_motion as eom 
import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
params = in_object.get_parameters_real_time_prop(path+'/', arg=1)

''' 
run the calculation protocoll: brief description:
    1. Create diverse objects for computing green function and energy
    2. Create uniform initialization of the wavefunction
    3. Propagate wavefunction for dt, compute Green and Energy
    4. Update psi_curr and evolve psi_curr again for dt
    5. Repeat steps 3. and 4. 
''' 

# create uniform initial wavefunction
wavefunc_object = h_wavef.wavefunctions(params=params)
psi_init = wavefunc_object.create_init_wavefunction('uniform') 

tic = time.perf_counter() # start timer

eom_object = eom.eom(params=params) # equations of motion object
eom_object.solve_for_fixed_params_real_time_prop(psi_init, path) # all objects for the calculation are calculated there!

toc = time.perf_counter() # end timer
print("\nExecution time = ", (toc-tic)/60, "min")