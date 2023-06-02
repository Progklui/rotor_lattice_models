import numpy as np
import os, sys, gc

path_main = os.path.dirname(__file__)
sys.path.append(path_main)

'''
    - Goal of the code: calculation of the wavefunction densities and/or polaron size for every time step
    - Call: python3 a8_wavefunctions_real_time_propagation.py PATH_TO_CALCULATION_INPUT_FILE PATH_TO_ANALYSIS_INPUT_FILE

    - Philosophy: 
        - Specify the calculation and analysis details in two input files
        - Analysis input file: allows to choose just a small subgrid (symmetric around the impurity, i.e. the lattice center)

    - Main output: rotor densities and/or polaron sizes for every time step
'''

# import user-defined classes 
import class_handle_input as h_in
import class_visualization as vis
import class_mass_size as mass_size

in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

# MAIN PART
n, M, Mx, My, B, tx, ty, V_0, qx, qy, time_steps, dt = in_object.get_parameters_real_time_prop(path_main=path_main+'/', arg=1)
chosen_Mx, chosen_My, plot_conf, plot_size = in_object.get_parameters_real_time_prop_analysis(path_main=path_main+'/', arg=2)

V_0_pool = np.array([V_0])

# read object
in_object_g = h_in.green_function(Mx=Mx, My=My, B=B, V_0=V_0, tx=tx, ty=ty, qx=qx, qy=qy, n=n, dt=dt, time_steps=time_steps)
folder_name_g, file_name_green = in_object_g.result_folder_structure_real_time_prop(path_main) # get the folder structure for results
folder_name_w, file_name_wavefunction = in_object_g.wavefunction_folder_structure_real_time_prop(path_main) # get the folder structure for wavefunctions

# calculation objects
size_object = mass_size.polaron_size(Mx=Mx, My=My, B=B, V_0=V_0_pool, tx=tx, ty=ty, qx=qx, qy=qy, n=n, dt=dt, tol=time_steps)
plot_object = vis.polaron_size(Mx=Mx, My=My, B=B, V_0=V_0_pool, tx=tx, ty=ty, qx=qx, qy=qy, n=n, dt=dt, tol=time_steps)

plot_object = vis.configurations(Mx=Mx, My=My, B=B, tx=tx, ty=ty, qx=qx, qy=qy, n=n, dt=dt, tol=time_steps, Vmin=V_0, Vmax=V_0)

# loop through all configurations
for i in range(time_steps):
    psi_rotors = np.load(folder_name_w+file_name_wavefunction+str(i)+'.npy').reshape(My,Mx,n)
    # Note: Vmin should be Vmax in these calculations!, no additional check is made here - user fault in case!
    print('Printing time step ', i)
    if plot_conf == 'y':
        plot_object.plot_configuration_real_time(psi_rotors, V_0, i, chosen_My, chosen_Mx, "real", path_main)
    if plot_size == 'y':
        sigma = size_object.calc_polaron_size(psi_rotors, i, '1')
        sigma = sigma[int((My-chosen_My)/2):int((My+chosen_My)/2), int((Mx-chosen_Mx)/2):int((Mx+chosen_Mx)/2)]
        plot_object.plot_polaron_size_real_time(sigma, V_0, i, 'scan_dir', path_main)

        del sigma 
        gc.collect()