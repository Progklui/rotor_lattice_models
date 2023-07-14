import numpy as np
from scipy.integrate import solve_ivp

import os, sys, csv, json

path = os.path.dirname(__file__) 
sys.path.append(path)

'''
TODO: document this class
'''
class params:
    def __init__(self, on_cluster):
        self.on_cluster = on_cluster

    def get_file_name(self, path_main, folder_name, file_name): # mainly used for storing the result in a consistent manner at the end of the calculation
        #path = path_main # os.path.join(os.path.dirname(__file__), folder_name)
        try: os.makedirs(path_main+folder_name)
        except FileExistsError: pass
        return path_main+folder_name+file_name # os.path.join(os.path.dirname(__file__), folder_name+file_name)

    def folder_structure_pot_crossing_scan(self,M,B,tx,ty,V1st,Vjump,potential_points):
        folder_name = 'matrix_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(B)+'_tx_'+str(tx)+'_ty_'+str(ty)\
            +'_V1st_'+str(V1st)+'_Vjump_'+str(Vjump)+'_Vnumber_'+str(potential_points)+'/'
        return folder_name
    
    def file_name_pot_crossing_scan(self,qx,qy,direction,init,init_repeat):
        file_name = 'psi_rotors_2d_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_'+str(direction)+'_init_'+str(init)+'_init_repeat_'+str(init_repeat)
        return file_name

    def get_file_path(self, arg):
        try:
            argument = sys.argv[int(arg)]
            if argument == "-h" or argument == "h" or argument == "-help" or argument == "help": print(" "); print("Use this argument structure: [PATH]"); print(" "); quit()
            else: file_path = argument
            print(" "); print("Verify Path: ", file_path); print(' ')
        except:
            print(" "); print("Verify Path: ", file_path); print(' ')
            pass

        if self.on_cluster == True: return file_path
        else:
            accept = input('Accept (y/n)? '); print(' ')
            if accept == 'y': return file_path
            else: exit()

    def get_parameters(self, path_main, arg):
        #path_main = os.path.dirname(os.path.abspath(__file__))+"/"
        file_path = self.get_file_path(arg)
        print('Current settings:'); print(' ')
        with open(path_main+file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            for row in spamreader:
                identifier = row[0]
                value = row[1].replace(" ", "")
                if identifier == "n": n = int(value); print('n    =', n) # grid size of the angle
                elif identifier == "M": M = int(value); print('M    =', M); Mx=int(M**0.5); My=int(M**0.5) # number of rotors - in 2D should be a square of an (even) number
                elif identifier == "Mx": Mx = int(value); print('Mx   =', Mx) # number of rotors - in 2D should be a square of an (even) number
                elif identifier == "My": My = int(value); print('My   =', My) # number of rotors - in 2D should be a square of an (even) number
                elif identifier == "B": B = float(value); print('B    =', B) # rotational energy of rotors
                elif identifier == "tx": tx = float(value); print('tx   =', tx) # tunneling along columns
                elif identifier == "ty": ty = float(value); print('ty   =', ty); print(' ') # tunneling along rows
                elif identifier == "pot_points": potential_points = int(value); print('#poi.=', potential_points) # number of potential points to calculate the wave functions
                elif identifier == "V1st": Vmin = float(value); print('V1st =', Vmin) 
                elif identifier == "Vjump": Vmax = float(value); print('Vjump =', Vmax) # maximum potential - should be larger than the largest tunneling rate?!
                elif identifier == "Vback": Vback = float(value); print('Vback =', Vback) # maximum potential - should be larger than the largest tunneling rate?!
                elif identifier == "pot_points_back": pot_points_back = int(value); print('#poi. back =', pot_points_back) # maximum potential - should be larger than the largest tunneling rate?!
                elif identifier == "scan_dir": scan_dir = str(value); print('Scan direction: '+scan_dir); print(' ') # forward/backward: forward scans from Vmin to Vmax, backward from Vmax to Vmin
                elif identifier == "qx": qx = float(value); print('qx   =', qx) # column momentum of electron
                elif identifier == "qy": qy = float(value); print('qy   =', qy); print(' ') # row momentum of electron
                elif identifier == "tol": tol = float(value); print('tol  =', tol) # for convergence - 1e-7 already sufficient for most qualitative behaviour (fast), e.g. 1e-12 runs significantly longer
                elif identifier == "dt": dt = float(value); print('dt   =', dt); print(' ') # time evolution
                elif identifier == "init": init = str(value); print('init =', init) # choice of initialization
                elif identifier == "init_repeat": init_rep = int(value); print('init repeat =', init_rep) # how often the run should be done
                elif identifier == "use_previous_init": use_previous = str(value); print('use previous init =', use_previous); print(' ') # whether to use previous solution as input to next
        try:
            init
            init_rep
            use_previous
        except NameError:
            init = None
            init_rep = None
            use_previous = 'true'

        if self.on_cluster == True: return n, M, Mx, My, B, tx, ty, potential_points, Vmin, Vmax, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous
        else:
            accept = input('Accept (y/n)? ')
            if accept == 'y': return n, M, Mx, My, B, tx, ty, potential_points, Vmin, Vmax, Vback, pot_points_back, qx, qy, tol, dt, scan_dir, init, init_rep, use_previous
            else: exit()

    def get_parameters_real_time_prop_analysis(self, path_main, arg):
        #path_main = os.path.dirname(os.path.abspath(__file__))+"/"
        file_path = self.get_file_path(arg)
        print('Current settings:'); print(' ')
        with open(path_main+file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            for row in spamreader:
                identifier = row[0]
                value = row[1].replace(" ", "")
                if identifier == "Mx_select": chosen_Mx = int(value); print('Mx chosen =', chosen_Mx) # number of rotors to display in Mx
                elif identifier == "My_select": chosen_My = int(value); print('My chosen =', chosen_My); print(' ') # number of rotors to display in My
                elif identifier == "plot_densities": plot_conf = str(value); print('Plot densities (y/n)?', plot_conf) # choose whether to plot rotor densities
                elif identifier == "plot_polaron_size": plot_size = str(value); print('Plot polaron size (y/n)?', plot_size); print(' ') # choose whether to plot polaron size

    
        if self.on_cluster == True: return chosen_Mx, chosen_My, plot_conf, plot_size
        else:
            accept = input('Accept (y/n)? ')
            print(' ')
            if accept == 'y': return chosen_Mx, chosen_My, plot_conf, plot_size
            else: exit()

    def get_parameters_real_time_prop(self, path_main, arg):
        print('Current settings:\n')

        # read the dictionary file
        file_path = self.get_file_path(arg)
        with open(path_main+file_path) as file:
            data = file.read()
        param_dict = json.loads(data)

        # print the parameters
        for key, value in param_dict.items(): 
            print(key,'=',value)

        # return the dictionary
        if self.on_cluster == True: return param_dict
        else:
            accept = input('\nAccept (y/n)? ')
            print(' ')
            if accept == 'y': return param_dict
            else: exit()

    def get_parameters_imag_time_prop(self, path_main, arg):
        print('Current settings:\n')

        # read the dictionary file
        file_path = self.get_file_path(arg)
        with open(path_main+file_path) as file:
            data = file.read()
        param_dict = json.loads(data)

        # print the parameters
        for key, value in param_dict.items(): 
            print(key,'=',value)

        # return the dictionary
        if self.on_cluster == True: return param_dict
        else:
            accept = input('\nAccept (y/n)? ')
            print(' ')
            if accept == 'y': return param_dict
            else: exit()

    def get_parameters_coupling_of_states(self, path_main, arg):
        print('Current settings:\n')

        # read the dictionary file
        file_path = self.get_file_path(arg)
        with open(path_main+file_path) as file:
            data = file.read()
        param_dict = json.loads(data)

        # print the parameters
        for key, value in param_dict.items(): 
            print(key,'=',value)

        # return the dictionary
        if self.on_cluster == True: return param_dict
        else:
            accept = input('\nAccept (y/n)? ')
            print(' ')
            if accept == 'y': return param_dict
            else: exit()

    def get_parameters_coupling_of_states(self, path_main, arg):
        #path_main = os.path.dirname(os.path.abspath(__file__))+"/"
        file_path = self.get_file_path(arg)
        print('Current settings:'); print(' ')
        with open(path_main+file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            for row in spamreader:
                identifier = row[0]
                value = row[1].replace(" ", "")
                if identifier == "pot_points": potential_points = int(value); print('#poi.=', potential_points) # number of potential points to calculate the wave functions
                elif identifier == "Vmin": Vmin = float(value); print('Vmin =', Vmin) 
                elif identifier == "Vmax": Vmax = float(value); print('Vmax =', Vmax); print(' ') # maximum potential - should be larger than the largest tunneling rate?!
                elif identifier == "n_states": n_states = int(value); print('# states =', n_states); print(' ') # number of states
                elif identifier == "path1": path1 = str(value); print('Path state 1:', path1) # Path state 1
                elif identifier == "path2": path2 = str(value); print('Path state 2:', path2) # Path state 2
                elif identifier == "path3": path3 = str(value); print('Path state 3:', path3) # Path state 3
                elif identifier == "path4": path4 = str(value); print('Path state 4:', path4); print(' ') # Path state 4 (could be a dummy path if n_states < 4)
        if self.on_cluster == True: return potential_points, Vmin, Vmax, n_states, path1, path2, path3, path4
        else:
            accept = input('Accept (y/n)? ')
            if accept == 'y': return potential_points, Vmin, Vmax, n_states, path1, path2, path3, path4
            else: exit()

class coupl_states:
    def __init__(self, params_calc, params_wfs):
        self.param_calc_dict = params_calc
        self.param_wfs_dict = params_wfs
        self.Mx  = int(params_calc['Mx'])
        self.My  = int(params_calc['My'])
        self.M   = int(params_calc['Mx']*params_calc['My'])
        self.B   = float(params_calc['B'])
        self.V_0 = 0 if isinstance(params_calc['V_0'], list) == True else float(params_calc['V_0'])
        self.tx  = float(params_calc['tx'])
        self.ty  = float(params_calc['ty'])
        self.qx  = int(params_calc['qx'])
        self.qy  = int(params_calc['qy'])
        self.n   = int(params_calc['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
        self.dt  = float(params_calc['dt'])
        self.tol = int(params_calc['tol'])
        self.n_states = params_wfs['n_states']

    def get_wavefunctions_per_interaction(self, path_main, V_0):
        psi_arr = [] 
        q_arr = []
        for i in range(self.n_states):
            descriptor_string = 'path'+str(i+1)
            wf_string = self.param_wfs_dict[descriptor_string] + str(V_0)+'.npy'

            wavefunc = np.load(path_main+wf_string).reshape(self.My, self.Mx, self.n)
            psi_arr.append(wavefunc)
            
            q = np.array([0,0])
            q_arr.append(q)

        return psi_arr, q_arr

    def energy_results_coupling_of_states(self, path_main):
        V_0_array = np.array(self.param_calc_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/coupling_of_states/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        energies_file_name = 'energies_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'

        return folder_name, energies_file_name

    def trans_probs_results_coupling_of_states(self, state_i, path_main):
        V_0_array = np.array(self.param_calc_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/coupling_of_states/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        probs_file_name = 'transition_probs_2d_state_'+str(state_i)+'_MRCI_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'
        
        return folder_name, probs_file_name
    
    def matrices_results_coupling_of_states(self, path_main):
        V_0_array = np.array(self.param_calc_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/coupling_of_states/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        heff_file_name = 'heff_2d_state_MRCI_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_V0_'
        s_overlap_file_name = 's_overlap_2d_state_MRCI_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_V0_'
        
        return folder_name, heff_file_name, s_overlap_file_name
    
    def store_matrices(self, V_0, h_eff, s_overlap, path_main):
        folder_name, heff_file_name, s_overlap_file_name = self.matrices_results_coupling_of_states(path_main)
        
        np.savetxt(folder_name+heff_file_name+str(V_0)+'.out', (h_eff))
        np.savetxt(folder_name+s_overlap_file_name+str(V_0)+'.out', (s_overlap))

    def store_matrices_during_computation(self, V_0, E_list, S_list, path_main):
        folder_name, heff_file_name, s_overlap_file_name = self.matrices_results_coupling_of_states(path_main)

        heff_file_name = heff_file_name+str(V_0)+'_during_comp.out'
        s_overlap_file_name = s_overlap_file_name+str(V_0)+'_during_comp.out'

        with open(folder_name+heff_file_name, 'a') as h_eff_file:
            write_string = ''
            for i in range(len(E_list)):
                write_string += str(E_list[i])+' '
            write_string += '\n'

            h_eff_file.write(write_string)

        with open(folder_name+s_overlap_file_name, 'a') as s_file:
            write_string = ''
            for i in range(len(S_list)):
                write_string += str(S_list[i])+' '
            write_string += '\n'

            s_file.write(write_string)

    def store_energies(self, V_0, E, path_main):
        folder_name, energies_file_name = self.energy_results_coupling_of_states(path_main)

        with open(folder_name+energies_file_name, 'a') as energy_file:
            write_string = str(V_0)
            for i in range(self.n_states):
                write_string += ' '+str(E[i])
            write_string += '\n'

            energy_file.write(write_string)

    def store_transition_probabilities(self, V_0, trans_probs_list, path_main):
        for state_i in range(self.n_states):
            folder_name, probs_file_name = self.trans_probs_results_coupling_of_states(state_i+1, path_main)
            
            trans_probs = trans_probs_list[state_i]
            
            with open(folder_name+probs_file_name, 'a') as prob_file:
                write_string = str(V_0)
                for i in range(self.n_states):
                    write_string += ' '+str(trans_probs[i])
                write_string += '\n'
                prob_file.write(write_string)

'''
TODO: document this class
'''
class imag_time:
    def __init__(self, params):
        self.param_dict = params
        self.Mx  = int(params['Mx'])
        self.My  = int(params['My'])
        self.M   = int(params['Mx']*params['My'])
        self.B   = float(params['B'])
        self.V_0 = 0 if isinstance(params['V_0'], list) == True else float(params['V_0'])
        self.tx  = float(params['tx'])
        self.ty  = float(params['ty'])
        self.qx  = int(params['qx'])
        self.qy  = int(params['qy'])
        self.n   = int(params['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
        self.dt  = float(params['dt'])
        self.tol = int(params['tol'])

    def wavefunction_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/matrix_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'psi_rotors_2d_imag_time_prop_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_dt_'+str(self.dt)+'_init_'+self.param_dict['init_choice']+'_V0_'
        return folder_name, file_name

    def energy_results_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/energies/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'energies_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'

        return folder_name, file_name
    
    def energy_results_folder_structure_imag_time_during_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/energies/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'energies_during_propagation_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'

        return folder_name, file_name
    
    def t_deriv_energy_results_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/energies/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'dE_dt_of_energies_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'

        return folder_name, file_name
    
    def t_deriv_energy_results_folder_structure_imag_time_during_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/energies/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'dE_dt_of_energies_during_prop_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'

        return folder_name, file_name
    
    def green_overlap_results_folder_structure_imag_time_during_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/energies/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'overlap_with_init_during_prop_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'

        return folder_name, file_name
    
    def polaron_size_results_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/polaron_size/'+'init_'+self.param_dict['init_choice']+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'pol_size_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_V0_'

        return folder_name, file_name
    
    def plot_rotor_density_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/rotor_densities/'+'init_'+self.param_dict['init_choice']+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'rotor_density_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_V0_'
        
        return folder_name, file_name
    
    def plot_rotor_phase_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/rotor_phases/'+'init_'+self.param_dict['init_choice']+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'rotor_phases_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_V0_'
        
        return folder_name, file_name
    
    def save_energies_during_prop(self, iter, V_0, E, path_main):
        folder_name_e, file_name_energies = self.energy_results_folder_structure_imag_time_during_prop(path_main=path_main)
        with open(folder_name_e+file_name_energies, 'a') as energy_file:
            write_string = str(iter)
            write_string += ' '+str(V_0)
            for i in range(len(E)):
                write_string += ' '+str(E[i])
            write_string += '\n'
            energy_file.write(write_string)
        return
    
    def save_dE_dt_during_prop(self, iter, V_0, dE_dtx, dE_dty, path_main):
        folder_name_e, file_name_energies = self.t_deriv_energy_results_folder_structure_imag_time_during_prop(path_main=path_main)
        with open(folder_name_e+file_name_energies, 'a') as de_dt_file:
            write_string = str(iter)+' '+str(V_0)+' '+str(dE_dtx)+' '+str(dE_dty)+'\n'
            de_dt_file.write(write_string)
        return
    
    def save_green_func_during_prop(self, iter, V_0, green, path_main):
        folder_name_e, file_name_energies = self.green_overlap_results_folder_structure_imag_time_during_prop(path_main=path_main)
        with open(folder_name_e+file_name_energies, 'a') as green_file:
            write_string = str(iter)+' '+str(V_0)+' '+str(green)+'\n'
            green_file.write(write_string)
        return
    
    def save_energies(self, V_0, E, path_main):
        folder_name_e, file_name_energies = self.energy_results_folder_structure_imag_time_prop(path_main=path_main)
        with open(folder_name_e+file_name_energies, 'a') as energy_file:
            write_string = str(V_0)
            for i in range(len(E)):
                write_string += ' '+str(E[i])
            write_string += '\n'
            energy_file.write(write_string)
        return

    def save_dE_dt(self, V_0, dE_dtx, dE_dty, path_main):
        folder_name_de_dt, file_name_de_dt = self.t_deriv_energy_results_folder_structure_imag_time_prop(path_main=path_main)
        with open(folder_name_de_dt+file_name_de_dt, 'a') as de_dt_file:
            write_string = str(V_0)+' '+str(dE_dtx)+' '+str(dE_dty)+'\n'
            de_dt_file.write(write_string)
        return
    
    def save_polaron_size(self, V_0, sigma, path_main):
        folder_name_p, file_name_size = self.polaron_size_results_folder_structure_imag_time_prop(path_main=path_main)
        np.savetxt(folder_name_p+file_name_size+str(V_0)+'.out', (sigma))
        return 
    
    def save_densities_phases(self, densities, phases, V_0, path_main):
        folder_name_d, file_name_d = self.plot_rotor_density_folder_structure_imag_time_prop(path_main)
        np.save(folder_name_d+file_name_d+str(V_0), densities)

        folder_name_p, file_name_p = self.plot_rotor_phase_folder_structure_imag_time_prop(path_main)
        np.save(folder_name_p+file_name_p+str(V_0), phases)
        return 

class imag_time_conv:
    def __init__(self, params):
        self.param_dict = params
        self.Mx  = int(params['Mx'])
        self.My  = int(params['My'])
        self.M   = int(params['Mx']*params['My'])
        self.B   = float(params['B'])
        self.V_0 = 0 if isinstance(params['V_0'], list) == True else float(params['V_0'])
        self.tx  = float(params['tx'])
        self.ty  = float(params['ty'])
        self.qx  = int(params['qx'])
        self.qy  = int(params['qy'])
        self.n   = int(params['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
        self.dt  = float(params['dt'])
        self.tol = int(params['tol'])

    def wavefunction_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/matrix_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'psi_rotors_2d_imag_time_prop_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_dt_'+str(self.dt)+'_init_'+self.param_dict['init_choice']+'_V0_'
        return folder_name, file_name

    def energy_results_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/energies/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'energies_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'

        return folder_name, file_name
    
    def t_deriv_energy_results_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/energies/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'dE_dt_of_energies_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'.out'

        return folder_name, file_name
    
    def polaron_size_results_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/polaron_size/'+'init_'+self.param_dict['init_choice']+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'pol_size_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_V0_'

        return folder_name, file_name
    
    def plot_rotor_density_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/rotor_densities/'+'init_'+self.param_dict['init_choice']+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'rotor_density_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_V0_'
        
        return folder_name, file_name
    
    def plot_rotor_phase_folder_structure_imag_time_prop(self, path_main):
        V_0_array = np.array(self.param_dict['V_0'], dtype=float)
        V_min = np.min(V_0_array)
        V_max = np.max(V_0_array)

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'/rotor_phases/'+'init_'+self.param_dict['init_choice']+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'rotor_phases_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
                +str(self.ty)+'_Vmin_'+str(V_min)+'_Vmax_'+str(V_max)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_init_'+self.param_dict['init_choice']\
                +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_V0_'
        
        return folder_name, file_name
    
    def save_energies(self, V_0, E, path_main):
        folder_name_e, file_name_energies = self.energy_results_folder_structure_imag_time_prop(path_main=path_main)
        with open(folder_name_e+file_name_energies, 'a') as energy_file:
            write_string = str(V_0)
            for i in range(len(E)):
                write_string += ' '+str(E[i])
            write_string += '\n'
            energy_file.write(write_string)
        return

    def save_dE_dt(self, V_0, dE_dtx, dE_dty, path_main):
        folder_name_de_dt, file_name_de_dt = self.t_deriv_energy_results_folder_structure_imag_time_prop(path_main=path_main)
        with open(folder_name_de_dt+file_name_de_dt, 'a') as de_dt_file:
            write_string = str(V_0)+' '+str(dE_dtx)+' '+str(dE_dty)+'\n'
            de_dt_file.write(write_string)
        return
    
    def save_polaron_size(self, V_0, sigma, path_main):
        folder_name_p, file_name_size = self.polaron_size_results_folder_structure_imag_time_prop(path_main=path_main)
        np.savetxt(folder_name_p+file_name_size+str(V_0)+'.out', (sigma))
        return 
    
    def save_densities_phases(self, densities, phases, V_0, path_main):
        folder_name_d, file_name_d = self.plot_rotor_density_folder_structure_imag_time_prop(path_main)
        np.save(folder_name_d+file_name_d+str(V_0), densities)

        folder_name_p, file_name_p = self.plot_rotor_phase_folder_structure_imag_time_prop(path_main)
        np.save(folder_name_p+file_name_p+str(V_0), phases)
        return 
    
'''
TODO: document this class
'''
class real_time: 
    def __init__(self, params):
        self.param_dict = params
        self.Mx  = int(params['Mx'])
        self.My  = int(params['My'])
        self.M   = int(params['Mx']*params['My'])
        self.B   = float(params['B'])
        self.V_0 = float(params['V_0'])
        self.tx  = float(params['tx'])
        self.ty  = float(params['ty'])
        self.qx  = int(params['qx'])
        self.qy  = int(params['qy'])
        self.n   = int(params['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
        self.dt  = float(params['dt'])
        self.time_steps = int(params['time_steps'])

    def wavefunction_folder_structure_real_time_prop(self, path_main):
        folder_name = path_main+'/matrix_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_V0_'+str(self.V_0)+'/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'psi_rotors_2d_real_time_prop_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
            +str(self.ty)+'_V0_'+str(self.V_0)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_time_step_'
        return folder_name, file_name
    
    def result_folder_structure_real_time_prop(self, path_main):
        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
            +str(self.ty)+'_V0_'+str(self.V_0)+'/green_functions/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'green_function_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_time_steps_'+str(self.time_steps)+'_dt_'+str(self.dt)+'.out'
        
        return folder_name, file_name
    
    def t_deriv_energy_results_folder_structure_real_time_prop(self, path_main):
        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
            +str(self.ty)+'_V0_'+str(self.V_0)+'/green_functions/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'dE_dt_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_time_steps_'+str(self.time_steps)+'_dt_'+str(self.dt)+'.out'

        return folder_name, file_name
    
    def plot_rotor_density_folder_structure_real_time_prop(self, path_main, i):
        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
            +str(self.ty)+'_V0_'+str(self.V_0)+'/rotor_densities/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'rotor_density_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_time_steps_'+str(self.time_steps)+'_dt_'+str(self.dt)+'_iter_time_'+str(i)
        
        return folder_name, file_name
    
    def plot_rotor_phase_folder_structure_real_time_prop(self, path_main, i):
        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
            +str(self.ty)+'_V0_'+str(self.V_0)+'/rotor_phases/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'rotor_phases_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_time_steps_'+str(self.time_steps)+'_dt_'+str(self.dt)+'_iter_time_'+str(i)
        
        return folder_name, file_name
    
    def plot_polaron_size_folder_structure_real_time_prop(self, path_main, i):
        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
            +str(self.ty)+'_V0_'+str(self.V_0)+'/polaron_size/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'pol_size_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_time_steps_'+str(self.time_steps)+'_dt_'+str(self.dt)+'_iter_time_'+str(i)
        
        return folder_name, file_name
    
    def save_polaron_size(self, sigma, i, path_main):
        folder_name_p, file_name_size = self.plot_polaron_size_folder_structure_real_time_prop(path_main=path_main, i=i)
        np.savetxt(folder_name_p+file_name_size+'.out', (sigma))
        return 
    
    def save_energies(self, iter, E, path_main):
        folder_name_g, file_name_green = self.result_folder_structure_real_time_prop(path_main) # get the folder structure for results
        with open(folder_name_g+file_name_green, 'a') as green_f_file:
            write_string = str(iter)
            for i in range(len(E)):
                write_string += ' '+str(E[i])
            write_string += '\n'
            green_f_file.write(write_string)
        return
    
    def save_dE_dt(self, iter, dE_dtx, dE_dty, path_main):
        folder_name_de_dt, file_name_de_dt = self.t_deriv_energy_results_folder_structure_real_time_prop(path_main=path_main)
        with open(folder_name_de_dt+file_name_de_dt, 'a') as de_dt_file:
            write_string = str(iter)+' '+str(dE_dtx)+' '+str(dE_dty)+'\n'
            de_dt_file.write(write_string)
        return
    
    def save_densities_phases(self, densities, phases, iter, path_main):
        folder_name_d, file_name_d = self.plot_rotor_density_folder_structure_real_time_prop(path_main, iter)
        np.save(folder_name_d+file_name_d, densities)

        folder_name_p, file_name_p = self.plot_rotor_phase_folder_structure_real_time_prop(path_main, iter)
        np.save(folder_name_p+file_name_p, phases)
        return 