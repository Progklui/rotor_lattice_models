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
n, M, Mx, My, B, tx, ty, potential_points, V1st, Vjump, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path+'/', arg=1)

M = int(Mx*My) # safety - for some input scripts M might not be equal to Mx*My (backward compatibility!)

x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid

folder_name = in_object.folder_structure_pot_crossing_scan(M,B,tx,ty,V1st,Vjump,potential_points)
file_name_real_time_prop = in_object.file_name_real_time_propagation(qx,qy,'real_prop',init,init_rep)
file_name_for_ind_time_steps = in_object.get_file_name(path+'/', folder_name, file_name_real_time_prop)

eom_object = eom.eom(Mx=Mx, My=My, B=B, V_0=V1st, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol) # equations of motion object

''' 
run the calculation protocoll: brief description:
    1. Create diverse objects for computing green function and energy
    2. Create uniform initialization of the wavefunction
    3. Propagate wavefunction for dt, compute Green and Energy
    4. Update psi_curr and evolve psi_curr again for dt
    5. Repeat steps 3. and 4. 
''' 
tic = time.perf_counter() # start timer

eom_object.solve_for_fixed_coupling_real_time_prop(path, file_name_for_ind_time_steps) # all objects for the calculation are calculated there!

toc = time.perf_counter() # end timer
print("\nExecution time = ", (toc-tic)/60, "min")