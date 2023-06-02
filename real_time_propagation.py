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

in_object = h_in.params(on_cluster=True) # object for handling inputs from command line

# MAIN PART
n, M, Mx, My, B, tx, ty, V_0, qx, qy, time_steps, dt = in_object.get_parameters_real_time_prop(path+'/', arg=1)

''' 
run the calculation protocoll: brief description:
    1. Create diverse objects for computing green function and energy
    2. Create uniform initialization of the wavefunction
    3. Propagate wavefunction for dt, compute Green and Energy
    4. Update psi_curr and evolve psi_curr again for dt
    5. Repeat steps 3. and 4. 
''' 

tic = time.perf_counter() # start timer

eom_object = eom.eom(Mx=Mx, My=My, B=B, V_0=V_0, tx=tx, ty=ty, qx=qx, qy=qy, n=n, dt=dt, tol=1e-12) # equations of motion object
eom_object.solve_for_fixed_coupling_real_time_prop(path, time_steps) # all objects for the calculation are calculated there!

toc = time.perf_counter() # end timer
print("\nExecution time = ", (toc-tic)/60, "min")