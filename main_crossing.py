import numpy as np
import time, os, sys

path = os.path.dirname(__file__) 
sys.path.append(path)

import class_equations_of_motion as eom 
import class_energy as energy
import class_handle_input as h_in

def run_individual_calculation(eom_object, V1st, Vjump, V_0_pool_f, V_0_pool_b, psi_col, psi_rotors_f, psi_rotors_b, M, Mx, My, n, use_previous, opt_rand):
    tic = time.perf_counter() # start timer

    # compute the wave functions for the first and last potential
    eom_object.V_0  = V1st
    psi_rotors_f[0] = eom_object.solve_for_fixed_coupling_imag_time_prop(psi_col.copy()).reshape((My,Mx,n)).copy()
    eom_object.V_0  = Vjump
    psi_rotors_b[0] = eom_object.solve_for_fixed_coupling_imag_time_prop(psi_rotors_f[0].reshape((M*n)).copy()).reshape((My,Mx,n)).copy()

    if use_previous == 'false':
        for i in range(1,potential_points): # loop over all potential points
            eom_object.V_0 = V_0_pool_f[i]
            if opt_rand == 'false':
                psi_sol = eom_object.solve_for_fixed_coupling_imag_time_prop(psi_col.reshape((M*n)).copy())
            elif opt_rand == 'true':
                # initialize wave function 
                psi_col = eom_object.create_initialization('3') # init options: '0' - uniform, '1' - ferroelectric domain, '2' - small polaron, '3' - random
                psi_sol = eom_object.solve_for_fixed_coupling_imag_time_prop(psi_col.reshape((M*n)).copy())
            psi_rotors_f[i] = psi_sol.reshape((My,Mx,n))
    elif use_previous == 'true':
        for i in range(1,potential_points): # loop over all potential points
            eom_object.V_0 = V_0_pool_f[i]
            psi_col = eom_object.solve_for_fixed_coupling_imag_time_prop(psi_rotors_f[i-1].reshape((M*n)).copy())
            psi_rotors_f[i] = psi_col.reshape((My,Mx,n))

        for i in range(1,pot_points_back): # loop over all potential points
            eom_object.V_0 = V_0_pool_b[i]
            psi_col = eom_object.solve_for_fixed_coupling_imag_time_prop(psi_rotors_b[i-1].reshape((M*n)).copy())
            psi_rotors_b[i] = psi_col.reshape((My,Mx,n))

    toc = time.perf_counter() # end timer
    print("\n Execution time = ", (toc-tic)/60, "min")
    return psi_rotors_f, psi_rotors_b

in_object = h_in.params(on_cluster=True) # object for handling inputs from command line

# MAIN PART
n, M, Mx, My, B, tx, ty, potential_points, V1st, Vjump, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path+'/', arg=1)

M = int(Mx*My) # safety - for some input scripts M might not be equal to Mx*My (backward compatibility!)

# potentials for forward and backward scanning around the supposed phase transition
V_0_pool_f = np.linspace(V1st, Vjump, potential_points)
V_0_pool_b = np.linspace(Vjump, Vback, pot_points_back)

x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid
dx = x[1] - x[0]

eom_object = eom.eom(Mx=Mx, My=My, B=B, V_0=V1st, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol) # equations of motion object

# At the moment following logic: for deterministic initializations, just run the protocoll once, for random initializations run the protocoll more often, but just 
# store the energies
if init != '3':
    psi_rotors_f = np.zeros((potential_points,My,Mx,n),dtype=complex) # array to store the rotor wave functions for all couplings for forward scan
    psi_rotors_b = np.zeros((pot_points_back,My,Mx,n),dtype=complex) # array to store the rotor wave functions for all couplings for backward scan


    # initialize wave function
    psi_col = eom_object.create_initialization(init) # init options: '0' - uniform, '1' - ferroelectric domain, '2' - small polaron, '3' - random


    # run the calculation protocoll
    psi_rotors_f, psi_rotors_b = run_individual_calculation(eom_object, V1st, Vjump, V_0_pool_f, V_0_pool_b, psi_col, psi_rotors_f, psi_rotors_b, M, Mx, My, n, use_previous, 'false')

    # save rotor wavefunctions
    psi_rotors_f = psi_rotors_f.reshape(potential_points,M,n)
    psi_rotors_b = psi_rotors_b.reshape(pot_points_back,M,n)

    folder_name = in_object.folder_structure_pot_crossing_scan(M,B,tx,ty,V1st,Vjump,potential_points)
    file_name_f = in_object.file_name_pot_crossing_scan(qx,qy,'forward',init,init_rep)
    file_name_b = in_object.file_name_pot_crossing_scan(qx,qy,'backward',init,init_rep)

    np.save(in_object.get_file_name(path+'/', folder_name, file_name_f), psi_rotors_f) # save to numpy file format for later analysis
    np.save(in_object.get_file_name(path+'/', folder_name, file_name_b), psi_rotors_b) # save to numpy file format for later analysis
elif init == '3':
    energy_object = energy.energy(Mx=Mx, My=My, B=B, V_0=0, tx=tx, ty=ty,
                                  qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol) # create energy object
    
    E_col_f = np.zeros((len(V_0_pool_f), init_rep, 4), dtype=complex)
    E_col_b = np.zeros((len(V_0_pool_b), init_rep, 4), dtype=complex)

    for j in range(init_rep):
        psi_rotors_f = np.zeros((potential_points,My,Mx,n),dtype=complex) # array to store the rotor wave functions for all couplings for forward scan
        psi_rotors_b = np.zeros((pot_points_back,My,Mx,n),dtype=complex) # array to store the rotor wave functions for all couplings for backward scan

        # initialize wave function 
        psi_col = eom_object.create_initialization(init) # init options: '0' - uniform, '1' - ferroelectric domain, '2' - small polaron, '3' - random

        # run the calculation protocoll
        psi_rotors_f, psi_rotors_b = run_individual_calculation(eom_object, V1st, Vjump, V_0_pool_f, V_0_pool_b, psi_col, psi_rotors_f, psi_rotors_b, M, Mx, My, n, use_previous, 'true')
        
        # compute energies from forward scan
        for i in range(len(V_0_pool_f)):
            energy_object.V_0 = V_0_pool_f[i] # update coupling in energy object
            E_col_f[i,j] = np.asarray(energy_object.calc_energy(psi_rotors_f[i]))

        # compute energies from backward scan
        for i in range(len(V_0_pool_b)):
            energy_object.V_0 = V_0_pool_b[i] # update coupling in energy object
            E_col_b[i,j] = np.asarray(energy_object.calc_energy(psi_rotors_b[i])) # [0] was here before?

    print('\nV_0 =', V_0_pool_f)
    print('E_f =', E_col_f.T)
    print('\nV_0 =', V_0_pool_b)
    print('E_b =', E_col_b.T)