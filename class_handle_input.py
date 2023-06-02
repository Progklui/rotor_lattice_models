import numpy as np
from scipy.integrate import solve_ivp

import os, sys, csv

path = os.path.dirname(__file__) 
sys.path.append(path)

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
                elif identifier == "My": My = int(value); print('My   =', My); print(' ') # number of rotors - in 2D should be a square of an (even) number
                elif identifier == "B": B = float(value); print('B    =', B) # rotational energy of rotors
                elif identifier == "tx": tx = float(value); print('tx   =', tx) # tunneling along columns
                elif identifier == "ty": ty = float(value); print('ty   =', ty) # tunneling along rows
                elif identifier == "V_0": V_0 = float(value); print('V_0 =', V_0); print(' ')
                elif identifier == "qx": qx = float(value); print('qx   =', qx) # column momentum of electron
                elif identifier == "qy": qy = float(value); print('qy   =', qy); print(' ') # row momentum of electron
                elif identifier == "dt": dt = float(value); print('dt   =', dt);  # time step length
                elif identifier == "time_steps": time_steps = int(value); print('time steps =', time_steps); print(' ') # number of time steps

        if self.on_cluster == True: return n, M, Mx, My, B, tx, ty, V_0, qx, qy, time_steps, dt
        else:
            accept = input('Accept (y/n)? ')
            if accept == 'y': return n, M, Mx, My, B, tx, ty, V_0, qx, qy, time_steps, dt
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

class green_function:
    def __init__(self, Mx, My, B, V_0, tx, ty, qx, qy, n, dt, time_steps):
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.V_0 = V_0
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.dt  = dt
        self.time_steps = time_steps

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
    
    def plot_result_folder_structure_real_time_prop(self, path_main, i):
        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(int(self.Mx*self.My))+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'\
            +str(self.ty)+'_V0_'+str(self.V_0)+'/rotor_densities/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = 'rotor_density_2d_M_'+str(int(self.Mx*self.My))+'_Mx_'+str(self.Mx)+'_My_'+str(self.My)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_time_steps_'+str(self.time_steps)+'_dt_'+str(self.dt)+'_iter_time_'+str(i)
        
        return folder_name, file_name