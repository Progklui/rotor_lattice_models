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

    def file_name_real_time_propagation(self,qx,qy,direction,init,init_repeat):
        file_name = 'psi_rotors_2d_real_time_prop_qx_'+str(qx)+'_qy_'+str(qy)+'_scan_direction_'+str(direction)+'_init_'+str(init)+'_init_repeat_'+str(init_repeat)
        return file_name
    
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
    def __init__(self, Mx, My, B, V_0, tx, ty, qx, qy, n, x, dt, tol):
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.V_0 = V_0
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.x   = x
        self.dt  = dt
        self.tol = tol

    def get_file_structure(self, path_main):
        M = int(self.My*self.Mx) # compute total number of rotors - not 'trivial' for Mx \neq My

        folder_name = path_main+'/image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
                +'_Vmin_'+str(self.V_0)+'_Vmax_'+str(self.V_0)+'_complete/green_functions/'
        
        try: os.makedirs(folder_name)
        except FileExistsError: pass

        file_name = folder_name+'green_function_2d_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
                +'_total_iter_'+str(self.tol)+'_dt_'+str(self.dt)+'_scan.out'
        
        return folder_name, file_name