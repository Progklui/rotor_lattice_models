import numpy as np
import os, sys

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
n, M, Mx, My, B, tx, ty, potential_points1, Vmin1, Vmax1, Vback1, pot_points_back1, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous = in_object.get_parameters(path_main=path_main+'/', arg=1)

V_0_pool = np.linspace(Vmin1, Vmax1, potential_points1)
x  = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid

folder_name = 'matrix_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)+'_V1st_'+str(Vmin1)+'_Vjump_'+str(Vmax1)+'_Vnumber_'+str(potential_points1)+'/'

file_name    = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_forward'+'_init_1-1_init_repeat_1'
psi_rotors_f = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name)+'.npy').reshape(potential_points1, My, Mx, n)

file_name_b   = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_backward'+'_init_1-1_repeat_1'
psi_rotors_b  = np.load(in_object.get_file_name(path_main+'/', folder_name, file_name)+'.npy')[::-1].reshape(potential_points1, My, Mx, n)


size_object = mass_size.polaron_size(Mx=Mx, My=My, B=B, V_0=V_0_pool, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)
plot_object = vis.polaron_size(Mx=Mx, My=My, B=B, V_0=V_0_pool, tx=tx, ty=ty, qx=qx, qy=qy, n=n, x=x, dt=dt, tol=tol)


# Loop over the different analysis possibilities
print_choice = 's'
while print_choice != 'q':
    print_choice = input('\nShow polaron size (single/all/ray/quit) = (s/a/r/q)? '); print(' ')
    if print_choice == 's': 
        print('Possible V_0:', V_0_pool); print(' ')

        V_index = int(input('Select V_0 index: '))
        print('Selected potential V_0 =', V_0_pool[V_index])
        calc_choice = input('\nSelect calculation choice (1/2/3): '); print(' ')

        sigma = size_object.calc_polaron_size(psi_rotors_f, V_index, calc_choice)
        plot_object.plot_polaron_size(sigma, V_index, path_main+'/', hide_show=False)

    elif print_choice == 'a': 
        print('V_0:', V_0_pool)
        calc_choice = input('\nSelect calculation choice (1/2/3): '); print(' ')

        for i in range(len(V_0_pool)):
            sigma = size_object.calc_polaron_size(psi_rotors_f, i, calc_choice)
            plot_object.plot_polaron_size(sigma, i, path_main+'/', hide_show=True)

    elif print_choice == 'r':
        print('Possible V_0:', V_0_pool); print(' ')

        V_index = list(map(int, input('Select V_0 indices (enter comma separated list): ').split(',')))
        print('Selected potentials V_0 =', V_0_pool[V_index]) 
        calc_choice = input('\nSelect calculation choice (1/2/3): ')

        x_index = int(input('\nSpecify x-component of ray: '))
        y_index = int(input('Specify y-component of ray: '))
        ray_indices = np.array([x_index, y_index])

        x_index_s = int(input('\nSpecify x-component of start: '))
        y_index_s = int(input('Specify y-component of start: ')); print(' ')
        start_indices = np.array([x_index_s, y_index_s])

        # make a stupid test run to get hold of the expected size of the array - certainly there is a better way
        sigma = size_object.calc_polaron_size(psi_rotors_f, 0, calc_choice).T
        sigma_line = size_object.calc_polaron_size_along_ray(sigma, ray_indices, start_indices)

        sigma_line_tot = np.zeros((len(V_0_pool), len(sigma_line))) # len(sigma_line), array to store for the selected potentials

        j = 0
        for i in range(len(V_0_pool)):
            sigma = size_object.calc_polaron_size(psi_rotors_f, i, calc_choice).T
            sigma_line_tot[j] = size_object.calc_polaron_size_along_ray(sigma, ray_indices, start_indices)

            j += 1

        plot_object.plot_polaron_size_along_line(sigma_line_tot, sigma_line_tot, V_index, ray_indices, start_indices, path_main+'/')