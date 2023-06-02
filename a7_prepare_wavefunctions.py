import numpy as np
import scipy 

import matplotlib.pyplot as plt
import matplotlib.ticker

import os, sys, csv

path_main = os.path.dirname(__file__) 
sys.path.append(path_main)
#sys.path.append('/home/fkluiben/Documents/phd_ista/software_projects/rotation_1_lemeshko/rotor_lattice_2d_python')

'''
    - Goal of the code: calculation of the coupling between 2! states
    - Call: python3 a5_coupling_2_states.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - Computes an effective hamiltonian and diagonalizes it (=generalized eigenvalue problem, considering the overlap matrix )
        - Storing the result in separate text-file for separate plotting
        - Results are: energies and the eigenvectors

    - Main output: energies from diagonalized hamiltonian, eigenvectors
'''

# import user-defined classes
import class_equations_of_motion as eom 
import class_energy as energy
import class_handle_input as h_in
import class_preparation as prep 
import class_visualization as vis

# psi expected in form (My, Mx, n) - output in form (My_new, Mx_new, n)
def expand_wavefunction(Mx_new, My_new, Mx, My, n, psi):
    psi = psi.reshape((My,Mx,n)).copy()
    psi_new = n**(-0.5)*np.ones((int(My_new),int(Mx_new),n),dtype=complex)

    iter = 0
    for i in range(My_new):
        for j in range(Mx_new):
            if i >= int((My_new-My)/2) and i < int((My_new+My)/2):
                if j >= int((Mx_new-Mx)/2) and j < int((Mx_new+Mx)/2):
                    psi_new[(i+int(My_new/2))%My_new,(j+int(Mx_new/2))%Mx_new] = psi[(i+int(My/2))%My-int((My_new-My)/2)%My, (j+int(Mx/2))%Mx-int((Mx_new-Mx)/2)%Mx].copy()
                    #psi[(i+int(My/2))%My-int((My_new-My)/2), (j+int(Mx/2))%Mx-int((Mx_new-Mx)/2)].copy()
                    #print('here ', iter)
                    #print(i, i-int((My_new-My)/2))
                    #print(j, j-int((Mx_new-Mx)/2))
                    
                    iter += 1

    normalization_factor = (1.0/np.sqrt(np.sum(np.abs(psi_new.reshape((int(Mx_new*My_new),n)))**2,axis=1))).reshape(int(Mx_new*My_new),1)
    return (normalization_factor*psi_new.reshape((int(Mx_new*My_new),n))).reshape((int(My_new),int(Mx_new),n))

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
# the input script is needed to specify the general symmetry of the problem, i.e. here tx/ty/B/n, the rest of the parameters are not really needed!, they are an artefact!
n, M, Mx, My, B, tx, ty, potential_points1, Vmin1, Vmax1, Vback1, pot_points_back1, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path_main=path_main+'/', arg=1)

M = int(Mx*My) # safety - for some input scripts M might not be equal to Mx*My

V_0_pool_f1 = np.linspace(Vmin1, Vmax1, potential_points1)
V_0_pool_b1 = np.linspace(Vmax1, Vback1, pot_points_back1)[::-1]

x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid


print('\nOptions: \n')
print('a - Calculate analytic small polaron state')
print('b - Collect wavefunctions from multiple sources')

choice = input('\nSelect (a/b) : ')
if choice == 'a':
    # Comment: for the small polaron state there is no need to add rotor states to the wavefunctions, as it is analytically known and can thus be always adjusted!
    # initialize wave function in the small polaron state
    Vmin = int(input('\nV_min    = '))
    Vmax = int(input('V_max    = '))
    pot_points = int((input('# points = ')))
    V_array = np.linspace(Vmin, Vmax, pot_points)

    eom_object = eom.eom(Mx=Mx, My=My, B=B, V_0=0, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol) # create equations of motion object
    energy_object = energy.energy(Mx=Mx, My=My, B=B, V_0=0, tx=tx, ty=ty,
                                qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol) # create energy object
    
    psi_array = np.zeros((len(V_array),My,Mx,n), dtype=complex)
    energ_array = np.zeros((len(V_array),4), dtype=complex)
    for i in range(len(V_array)):
        # compute wavefunctions
        eom_object.V_0 = V_array[i] # update coupling in eom object
        psi = eom_object.create_initialization('4') # init options: '0' - uniform, '1' - ferroelectric domain, '2' - small polaron, '3' - random
        psi_array[i] = psi.reshape(My,Mx,n).copy()
        
        # compute energies
        energy_object.V_0 = V_array[i] # update coupling in energy object
        energ_array[i] = np.asarray(energy_object.calc_energy(psi.copy()))

    # save results to a text file output - plotting and later analysis can thus be easily outsourced
    folder_name = '/prepared_wavefunctions/tx_'+str(tx)+'_ty_'+str(ty)+'_M_'+str(M)+'_B_'+str(B)+'/'\
            +'Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'/energies/'
    file_name   = 'energ_small_polaron_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_qx_'+str(qx)+'_qy_'+str(qy)+'_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)\
            +'_tol_'+str(tol)+'_dt_'+str(dt)

    # saves energies in format (E_tot, E_transfer, E_rotational, E_coupling)
    np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'.out', np.transpose([V_array, \
            energ_array.T[0], energ_array.T[1], energ_array.T[2], energ_array.T[3]]), delimiter=' ') 
    
    folder_name = '/prepared_wavefunctions/tx_'+str(tx)+'_ty_'+str(ty)+'_M_'+str(M)+'_B_'+str(B)+'/'\
            +'Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'/wavefunctions/'
    file_name   = 'psi_small_polaron_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_qx_'+str(qx)+'_qy_'+str(qy)+'_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)\
            +'_tol_'+str(tol)+'_dt_'+str(dt)
    np.save(in_object.get_file_name(path_main, folder_name, file_name), psi_array) # save to numpy file format for later analysis

    print('\nV_0 =', V_array)
    print('E =', energ_array.T)

elif choice == 'b':
    file_number = int(input('\nHow many wavefunctions? '))

    psi_list = []
    V_list = []
    for i in range(file_number):
        print('\nEnter details:')
        Vmin = int(input('\nV_min    = '))
        Vmax = int(input('V_max    = '))
        pot_points = int((input('# points = ')))
        V_array = np.linspace(Vmin, Vmax, pot_points)

        folder_name = str(input('\nPath to wavefunction file: '))
        file_name = str(input('Wavefunction file name: '))

        f_or_backward = str(input('\nIs this a forward or backward scan (f/b)? '))
        if f_or_backward == 'f':
            psi_rotors = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name))
        elif f_or_backward == 'b':
            psi_rotors = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name))[::-1]

        energy_object = energy.energy(Mx=Mx, My=My, B=B, V_0=0, tx=tx, ty=ty,
                                qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol) # create energy object
    
        energ_array = np.zeros((len(V_array),4), dtype=complex)
        for i in range(len(V_array)):
            # compute energies
            energy_object.V_0 = V_array[i] # update coupling in energy object
            energ_array[i] = np.asarray(energy_object.calc_energy(psi_rotors[i].reshape(My,Mx,n).copy()))
        
        print('\nV_0 =', V_array)
        print('E =', energ_array.T)

        plt.plot(V_array, energ_array.T[0].real, marker='x')
        plt.show()
        
        V_chosen = np.array([float(value) for value in input('\nSelect the potential points to keep: ').split(',')])
        print('Selected V_0 =', V_chosen)
        
        expand = input('\nExpand wavefunctions (y/n)? ')

        if expand == 'y':
            Mx_new = int(input('\nNew Mx = '))
            My_new = int(input('New My = '))

        for i in range(len(V_array)):
            for j in range(len(V_chosen)):
                if V_array[i] == V_chosen[j]:
                    if expand == 'y':
                        psi_new = expand_wavefunction(Mx_new, My_new, Mx, My, n, psi_rotors[i].reshape(My,Mx,n).copy())
                        psi_list.append(psi_new)
                        V_list.append(V_array[i])
                    elif expand =='n':
                        psi_list.append(psi_rotors[i].reshape(My,Mx,n).copy())
                        V_list.append(V_array[i])

    V_array = np.array(V_list)
    psi_array = np.array(psi_list)
    
    energ_array = np.zeros((len(V_array),4), dtype=complex)
    for i in range(len(V_array)):
        # compute energies
        energy_object.V_0 = V_array[i] # update coupling in energy object
        if expand == 'y':
            energy_object.Mx = Mx_new
            energy_object.My = My_new
            M = int(Mx_new*My_new)
            energ_array[i] = np.asarray(energy_object.calc_energy(psi_array[i].reshape(My_new,Mx_new,n).copy()))
        else:
            energ_array[i] = np.asarray(energy_object.calc_energy(psi_array[i].reshape(My,Mx,n).copy()))

    print('\nSummary of the selection: ')
    print('\nV_0 =', V_array)
    print('E =', energ_array.T)
    
    plt.plot(V_array, energ_array.T[0].real, marker='x')
    plt.show()

    confirm = input('\nSave these results (y/n)? ')
    if confirm == 'y':
        # save results to a text file output - plotting and later analysis can thus be easily outsourced
        folder_name = '/prepared_wavefunctions/tx_'+str(tx)+'_ty_'+str(ty)+'_M_'+str(M)+'_B_'+str(B)+'/'\
                +'Vmin_'+str(V_array[0])+'_Vmax_'+str(V_array[len(V_array)-1])+'/energies/'
        file_name   = 'energ_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_qx_'+str(qx)+'_qy_'+str(qy)+'_Vmin_'+str(V_array[0])+'_Vmax_'+str(V_array[len(V_array)-1])\
                +'_tol_'+str(tol)+'_dt_'+str(dt)

        # saves energies in format (E_tot, E_transfer, E_rotational, E_coupling)
        np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'.out', np.transpose([V_array, \
                energ_array.T[0], energ_array.T[1], energ_array.T[2], energ_array.T[3]]), delimiter=' ') 
    

        folder_name = '/prepared_wavefunctions/tx_'+str(tx)+'_ty_'+str(ty)+'_M_'+str(M)+'_B_'+str(B)+'/'\
                        +'Vmin_'+str(V_array[0])+'_Vmax_'+str(V_array[len(V_array)-1])+'/wavefunctions/'
        file_name   = 'psi_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_qx_'+str(qx)+'_qy_'+str(qy)+'_Vmin_'+str(V_array[0])+'_Vmax_'+str(V_array[len(V_array)-1])\
                        +'_tol_'+str(tol)+'_dt_'+str(dt)
        np.save(in_object.get_file_name(path_main, folder_name, file_name), psi_array) # save to numpy file format for later analysis