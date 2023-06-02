import numpy as np
import os, sys, gc

path_main = os.path.dirname(__file__)
sys.path.append(path_main)
#sys.path.append('/home/fkluiben/Documents/phd_ista/software_projects/rotation_1_lemeshko/rotor_lattice_2d_python')

'''
    - Goal of the code: calculation of the polaron size 
    - Call: python3 a3_polaron_size.py PATH_TO_INPUT_FILE

    - Philosophy: 
        - This allows to quickly check whether the system size was close enough to therm. limit
        - Store the polaron size mesh (x,y array) - this enables nice plottings afterwards
        - Three modes: 
            - s (polaron size mesh for single potential)
            - a (polaron size mesh for all potentials)
            - r (polaron size along a ray)

    - Main output: polaron size mesh grids!
'''

# import user-defined classes 
import class_handle_input as h_in
import class_visualization as vis
import class_mass_size as mass_size

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
n, M, Mx, My, B, tx, ty, potential_points, Vmin, Vmax, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path_main=path_main+'/', arg=1)

V_0_pool = np.linspace(Vmin, Vmax, potential_points)
x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid

# read in wavefunction object from file path
folder_name = in_object.folder_structure_pot_crossing_scan(M,B,tx,ty,Vmin,Vmax,potential_points)
file_name  = in_object.file_name_real_time_propagation(qx,qy,'real_prop',init,init_rep)

size_object = mass_size.polaron_size(Mx=Mx, My=My, B=B, V_0=V_0_pool, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)
plot_object = vis.polaron_size(Mx=Mx, My=My, B=B, V_0=V_0_pool, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)

size_object = mass_size.polaron_size(Mx=Mx, My=My, B=B, V_0=V_0_pool, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)
plot_object = vis.configurations(Mx=Mx, My=My, B=B, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol, Vmin=Vmin, Vmax=Vmax)

# TOdo: implement here something that takes care of the lattice size selection, also plot polaron size

print(" ")
print('Choose how man rotors you want to display:\n')
chosen_My = int(input("M_y rotors = "))
chosen_Mx = int(input("M_x rotors = "))
calc_choice = input('\nSelect polaron size calculation choice (1/2/3): '); print(' ')
plot_conf = input('Plot densities (y/n)? ')
plot_size = input('Plot polaron size (y/n)? '); print(' ')
# loop through all configurations
for i in range(int(tol)):
    psi_rotors = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name)+'_time_step_'+str(i)+'.npy').reshape(My,Mx,n)
    # Note: Vmin should be Vmax in these calculations!, no additional check is made here - user fault in case!
    print('Printing time step ', i)
    if plot_conf == 'y':
        plot_object.plot_configuration_real_time(psi_rotors, Vmin, i, chosen_My, chosen_Mx, "real", path_main)
    if plot_size == 'y':
        sigma = size_object.calc_polaron_size(psi_rotors, i, calc_choice)
        sigma = sigma[int((My-chosen_My)/2):int((My+chosen_My)/2), int((Mx-chosen_Mx)/2):int((Mx+chosen_Mx)/2)]
        plot_object.plot_polaron_size_real_time(sigma, Vmin, i, scan_dir, path_main)

        del sigma 
        gc.collect()