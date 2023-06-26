import numpy as np
import scipy 

import time 

import os, sys, gc
path = os.path.dirname(__file__) 
sys.path.append(path)

import class_equations_of_motion as eom 

class wavefunctions:
    ''' Class for wavefunction creation
        ----
        
        ----
        Inputs: 
            params: dictionary that contains the class variables
        ----

        ----
        Class variables:
            n (int): length of angle grid
            Mx (int): number of rotors in x direction
            My (int): number of rotors in y direction
            M (int): Mx*My, total number of rotor
        ----

        ----
        Methods:
            self.create_init_wavefunction(phase): computes "approximate" psi (My,Mx,n) with symmetry specified by input variable 'phase'
        ----
    '''

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
        '''
            TODO: make external functional that creates the angle grid - such that this can be changed externally somehow!
        '''
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid

    def init_read_in_wf(self, path_to_file):
        '''
            Computes: reads in already computed wavefunction

            ----
            Inputs: 
                path to wavefunction file starting from the location of the source .py file
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction
            ----
        '''
        
        psi_init = np.load(path+'/'+path_to_file).reshape((self.My,self.Mx,self.n))
        return psi_init
    
    def init_uniform(self):
        '''
            Computes: uniform wavefunction

            ----
            Inputs: 
                None
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): uniform wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output uniform wavefunction
            ----
        '''

        psi_init = self.n**(-0.5)*np.ones((self.My,self.Mx,self.n),dtype=complex)
        return psi_init
    
    def init_ferro_domain(self, orientation):
        '''
            Computes: ferroelectric domain wall wavefunction

            ----
            Comment:
                the parametrization of the wavefunction is empirical!
            ----

            ----
            Inputs: 
                orientation (string): OPTIONS: 'vertical' or 'horizontal', i.e. orientation of domain wall
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): ferroelectric domain wall wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction
            ----
        '''

        psi_init = self.init_uniform() # create object

        if orientation == 'vertical':
            for i in range(self.My):
                # other option is to try with np.cos(0.5*x) and np.sin(...)

                # left column
                psi_init[i,self.Mx-1] = np.cos(self.x) + 0j
                psi_init[i,self.Mx-1][int(self.n/4):int(self.n/2)] = 0.01 + 0j
                psi_init[i,self.Mx-1][int(self.n/2):int(3*self.n/4)] = 0.01 + 0j

                # right column
                psi_init[i,0] = np.cos(self.x) + 0j
                psi_init[i,0][0:int(self.n/4)] = 0.01 + 0j
                psi_init[i,0][int(3*self.n/4):self.n] = 0.01 + 0j
                
        elif orientation == 'horizontal':
            for j in range(self.Mx):
                # top row
                psi_init[self.My-1,j] = np.sin(self.x) 
                psi_init[self.My-1,j][0:int(self.n/2)] = 0.01 

                # bottom row
                psi_init[0,j] = np.sin(self.x) 
                psi_init[0,j][int(self.n/2):self.n] = 0.01 
        return psi_init
    
    def init_small_polaron(self):
        '''
            Computes: analytic small polaron wavefunction

            ----
            Comments: 
                Here we generate the GS Mathieu functions with even n=0, although it would be possible to generate excited Mathieu states (ask Georgios)
                TODO: discuss whether we should consider excited Mathieu states?
            ----

            ----
            Inputs: 
                None
            ----
            
            ----
            Variables:
                mathieu_parameter (scalar): parameter of the mathieu equation
                y (dimension: n): mathieu function for the parameter mathieu_parameter and on x-axis
                yp (dimension: n): first derivative of mathieu function
                psi_init (3-dimensional: My,Mx,n): small polaron wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction
            ----
        '''

        mathieu_parameter = 2*self.V_0/self.B
        psi_init = self.init_uniform() # create object

        # bottom left
        y, yp = scipy.special.mathieu_cem(0, mathieu_parameter, (self.x+3*np.pi/4)/2*180/np.pi)
        psi_init[0,self.Mx-1] = y/np.sqrt(np.sum(y*y))
        
        # bottom right
        y, yp = scipy.special.mathieu_cem(0, mathieu_parameter, (self.x+np.pi/4)/2*180/np.pi)
        psi_init[0,0] = y/np.sqrt(np.sum(y*y))

        # top left
        y, yp = scipy.special.mathieu_cem(0, mathieu_parameter, (self.x-3*np.pi/4)/2*180/np.pi)
        psi_init[self.My-1,self.Mx-1] = y/np.sqrt(np.sum(y*y))

        # top right
        y, yp = scipy.special.mathieu_cem(0, mathieu_parameter, (self.x-np.pi/4)/2*180/np.pi)
        psi_init[self.My-1,0] = y/np.sqrt(np.sum(y*y))

        return psi_init
    
    def init_random(self):
        '''
            Computes: smooth random wavefunction

            ----
            Inputs: 
                None
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): small polaron wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction
            ----
        '''

        psi_init = self.init_uniform # create object

        for i in range(self.My):
            for j in range(self.Mx):
                H = 10
                rho = np.random.rand(1,H)*np.logspace(-0.5,-2.5,H)
                phi = np.random.rand(1,H)*2*np.pi

                # Accumulate r(t) over t=[0,2*pi]
                t = (2*np.pi/self.n)*np.arange(self.n) # np.linspace(0,2*np.pi,n)
                r = np.ones(len(t))
                for h in range(H):
                    r = r + rho[0][h]*np.ones(len(t))*np.sin(h*t+phi[0][h]*np.ones(len(t)))

                # Reconstruct x(t), y(t)
                x = r*np.cos(t)
                y = r*np.sin(t)

                psi_init[i,j] = r + r*1j # not entirely sure about the imaginary part here

        return psi_init
    
    def create_init_wavefunction(self, phase):
        '''
            Computes: initial wavefunctions, mainly for imag time propagation

            ----
            Inputs:
                phase (string): specified in input file, options:
                    - phase == 'uniform': Y_11
                    - phase == 'ferro_domain_vertical_wall': polarized domain wall states, vertically
                    - phase == 'ferro_domain_horizontal_wall': polarized domain wall states, horizontally
                    - phase == 'random': continous random wavefunctions
                    - phase == 'small_polaron': analytic Mathieu function for the 4 inner rotors
                    - phase == 'external': initialize with an external wavefunction 
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): array which functions are to be defined here
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction with the initialized symmetry
            ----
        '''

        if phase == 'uniform':
            psi_init = self.init_uniform()

        elif phase == 'ferro_domain_vertical_wall': 
            psi_init = self.init_ferro_domain('vertical')
                
        elif phase == 'ferro_domain_horizontal_wall': 
            psi_init = self.init_ferro_domain('horizontal')

        elif phase == 'small_polaron': 
            psi_init = self.init_small_polaron()

        elif phase == 'random': 
            psi_init = self.init_random()

        elif phase == 'external': 
            '''
                TODO: implement check whether the sizes are correct
            '''
            path_to_file = self.param_dict['path_to_input_wavefunction']
            psi_init = self.init_read_in_wf(path_to_file)

        else: # sanity check
            return 
        
        wfn_manip = wavefunc_operations(params=self.param_dict)
        psi_init = wfn_manip.normalize_wf(psi_init, shape=(self.My,self.Mx,self.n))
        return psi_init

class wavefunc_operations:
    ''' Class for elementary wavefunc operations
        ----
        
        ----
        Inputs: 
            params: dictionary that contains the class variables
        ----

        ----
        Class variables:
            n (int): length of angle grid
            Mx (int): number of rotors in x direction
            My (int): number of rotors in y direction
            M (int): Mx*My, total number of rotor
        ----

        ----
        Methods:
            self.normalization_factor_wf(psi): computes 1/norm = normalization factor for every individual rotor
            self.normalize_wf(psi, shape): outputs a w.f. in which every individual rotor is normalized
            self.reshape_one_dim(psi): reshapes psi to My*Mx*n
            self.reshape_two_dim(psi): reshapes psi to (My*Mx,n)
            self.reshape_three_dim(psi): reshapes psi to (My,Mx,n)
        ----
    '''    

    def __init__(self, params):
        self.param_dict = params
        self.Mx = int(params['Mx'])
        self.My = int(params['My'])
        self.M  = int(params['Mx']*params['My'])
        self.n  = int(params['n'])

    def normalization_factor_wf(self, psi):
        '''
            ----
            Description: computes the norm factor for every rotor
                (1) take abs(psi)**2 of every rotor
                (2) sum over the 2nd axis (axis=1), i.e. sum over angles
                (3) take the square and the inverse to get the norm-factor
                (4) this gives a (M,1) array, specifying the normalization factor for every rotor
            ----

            ----
            Inputs:
                psi (shape doesn't matter: max. 3-dimensional)
            ----
            
            ----
            Variables:
                norm2 (2-dimensional: (M,n)): norm2 of psi
                norm_sqrt (2-dimensional: (M,1)): sqrt of norm2, summed over angle n, i.e. sqrt of norm for every single rotor 
            ---- 
            
            ----
            Outputs:
                normalization_factor (shape: (My*Mx,1)): normalization factor for every rotor
            ----
        '''
        psi = self.reshape_two_dim(psi) 

        norm2 = np.abs(psi)**2
        norm_sqrt = np.sqrt(np.sum(norm2,axis=1)).reshape(self.M,1)

        normalization_factor = 1.0/norm_sqrt
        return normalization_factor
    
    def normalize_wf(self, psi, shape):
        ''' 
            ----
            Description: normalizes every single rotor
            ----

            ----
            Inputs:
                psi (shape doesn't matter: max. 3-dimensional): wavefunction to normalize
            ----

            ----
            Outputs:
                psi (shape as specified by input shape=(,,)): normalized wavefunction
            ----
        '''

        normalization_factor = self.normalization_factor_wf(psi) # 1./norm 
        psi = normalization_factor*self.reshape_two_dim(psi)
        return psi.reshape(shape)
    
    def reshape_one_dim(self, psi):
        '''
            ----
            Output:
                psi (1-dimensional (My*Mx*n))
            ----
        '''
        return psi.reshape((self.My*self.Mx*self.n))
    
    def reshape_two_dim(self, psi):
        '''
            ----
            Output:
                psi (2-dimensional (My*Mx,n))
            ----
        '''
        return psi.reshape((self.My*self.Mx,self.n))
    
    def reshape_three_dim(self, psi):
        '''
            ----
            Output:
                psi (3-dimensional (My,Mx,n))
            ----
        '''
        return psi.reshape((self.My,self.Mx,self.n))

    def calc_overlap(self, psi1, psi2):
        ''' 
            ----
            Description: total overlap of psi1 and psi2
            ----

            ----
            Inputs:
                psi1 (max. 3-dimensional, but dimension is checked)
                psi2 (max. 3-dimensional, but dimension is checked)
            ----

            ----
            Variables:
                psi1_conj: conjugate of psi1
                overlap (scalar, dtype=complex): total overlap
            ----
            Output:
                overlap
            ----
        '''

        psi1 = self.reshape_three_dim(psi1) # for safety, to ensure that it is always of same shape
        psi2 = self.reshape_three_dim(psi2) # psi2.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape

        psi1_conj = np.conjugate(psi1)

        overlap = 1 + 0j
        for k in range(self.My): 
            for p in range(self.Mx):
                overlap *= np.sum(psi1_conj[k,p]*psi2[k,p])
        return overlap
    
    def single_rotor_overlap(self, psi1, psi2):
        ''' 
            ----
            Description: single rotor overlaps of psi1 and psi2 
            ----

            ----
            Inputs:
                psi1 (max. 3-dimensional, but dimension is checked)
                psi2 (max. 3-dimensional, but dimension is checked)
            ----

            ----
            Variables:
                psi1_conj: conjugate of psi1
            ----

            ----
            Output:
                overlap (1-dimensional (M,)): overlap of the M rotors in psi1 and psi2 
            ----
        '''

        psi1 = self.reshape_two_dim(psi1) # now a (M,n) object
        psi2 = self.reshape_two_dim(psi2) # now a (M,n) object

        psi1_conj = np.conjugate(psi1)

        '''
        sum over angle axis
        '''
        overlap = np.sum(psi1_conj*psi2, axis=1) 
        return overlap
    
    def cut_out_rotor_region(self, psi, chosen_My, chosen_Mx):
        ''' 
            ----
            Description: 
                - Cuts out a specified number of rotors, given by chosen_My and chosen_Mx
                - Is the contrast to the function add_rotors_to_wavefunction(...)
            ----

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n)): input wavefunction
                chosen_My (int, scalar): the chosen number of y rotors
                chosen_Mx (int, scalar): the chosen number of x rotors
            ----

            ----
            Outputs:
                psi_new (3-dimensional: (chosen_My, chosen_Mx, n)): wavefunction with smaller number of rotors
            ----
        '''

        psi_new = np.zeros((chosen_My,chosen_Mx,self.n), dtype=complex)

        for i in range(self.My):
            for j in range(self.Mx):
                border_i_left  = int((self.My-chosen_My)/2)
                border_i_right = int((self.My+chosen_My)/2)

                border_j_left  = int((self.Mx-chosen_Mx)/2)
                border_j_right = int((self.Mx+chosen_Mx)/2)

                if i >= border_i_left and i < border_i_right:
                    if j >= border_j_left and j < border_j_right:
                        psi_ind_i = (i+int(self.My/2))%self.My
                        psi_ind_j = (j+int(self.Mx/2))%self.Mx

                        psi_new[i-border_i_left, j-border_j_left] = psi[psi_ind_i,psi_ind_j]
                        
        return psi_new
    
    def individual_rotor_density(self, psi, chosen_My, chosen_Mx):
        ''' 
            ----
            Description: computes the density for every single rotor
            ----

            ----
            Inputs:
                psi (3-dimensional: (chosen_My,chosen_Mx,n)): input wavefunction
                chosen_My (int, scalar): the chosen number of y rotors
                chosen_Mx (int, scalar): the chosen number of x rotors
            ----

            ----
            Outputs:
                rotor_density (3-dimensional: (chosen_My, chosen_Mx, n)): rotor density for every individual rotor
            ----
        '''

        rotor_density = np.zeros((chosen_My,chosen_Mx,self.n), dtype=complex)

        for i in range(chosen_My):
            for j in range(chosen_Mx):
                #psi_ind_i = (i+int(chosen_My/2))%chosen_My
                #psi_ind_j = (j+int(chosen_Mx/2))%chosen_Mx

                ind_rotor_psi = psi[i,j]
                rotor_density[i,j] = (np.conjugate(ind_rotor_psi)*ind_rotor_psi).T 

        return rotor_density
    
    def individual_rotor_phase(self, psi, chosen_My, chosen_Mx):
        ''' 
            ----
            Description: computes the phase for every single rotor
            ----

            ----
            Comment: there is another way to compute the phase, instead of using the numpy function:
                sign_fac = np.sign(ind_rotor_psi.imag) # an (n,) object
                phase_without_sign = np.arccos(ind_rotor_psi.real/np.abs(ind_rotor_psi)) # an (n,) object
                
                phase = sign_fac*phase_without_sign
            ----
            Inputs:
                psi (3-dimensional: (chosen_My,chosen_Mx,n)): input wavefunction
                chosen_My (int, scalar): the chosen number of y rotors
                chosen_Mx (int, scalar): the chosen number of x rotors
            ----

            ----
            Outputs:
                rotor_pase (3-dimensional: (chosen_My, chosen_Mx, n)): rotor density for every individual rotor
            ----
        '''
        
        rotor_pase = np.zeros((chosen_My,chosen_Mx,self.n), dtype=complex)

        for i in range(chosen_My):
            for j in range(chosen_Mx):
                #psi_ind_i = (i+int(chosen_My/2))%chosen_My
                #psi_ind_j = (j+int(chosen_Mx/2))%chosen_Mx

                ind_rotor_psi = psi[i,j]

                phase = np.arctan2(ind_rotor_psi.imag,ind_rotor_psi.real) 

                rotor_pase[i,j] = phase

        return rotor_pase

    def add_rotors_to_wavefunction(self, psi):
        ''' 
            ----
            Description: 
                Adds uniformly oriented rotors to wavefunction - for coupling of state calculations necessary
            ----

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n)): wavefunction to expand
            ----

            ----
            Outputs:
                psi_new (3-dimensional: (My_new, Mx_new, n)): normalized wavefunction with bigger number of rotors
            ----
        '''

        Mx_new = self.param_dict['Mx_new']
        My_new = self.param_dict['My_new']

        psi = psi.reshape((self.My,self.Mx,self.n))
        psi_new = self.n**(-0.5)*np.ones((int(My_new),int(Mx_new),self.n),dtype=complex)

        for i in range(My_new):
            for j in range(Mx_new):
                if i >= int((My_new-self.My)/2) and i < int((My_new+self.My)/2):
                    if j >= int((Mx_new-self.Mx)/2) and j < int((Mx_new+self.Mx)/2):
                        psi_new[(i+int(My_new/2))%My_new,(j+int(Mx_new/2))%Mx_new] = psi[(i+int(self.My/2))%self.My-int((My_new-self.My)/2)%self.My, \
                                                                                         (j+int(self.Mx/2))%self.Mx-int((Mx_new-self.Mx)/2)%self.Mx]

        # update object Mx, My values for normalization
        self.Mx = Mx_new 
        self.My = My_new
        self.M  = int(Mx_new*My_new)

        psi_out = self.normalize_wf(psi_new, shape=(My_new,Mx_new,self.n))

        del psi_new
        gc.collect()
        # normalization_factor = (1.0/np.sqrt(np.sum(np.abs(psi_new.reshape((int(Mx_new*My_new),self.n)))**2,axis=1))).reshape(int(Mx_new*My_new),1)
        return psi_out #(normalization_factor*psi_new.reshape((int(Mx_new*My_new),self.n))).reshape((int(My_new),int(Mx_new),self.n))

    def expand_and_converge_wf(self, Mx_new_list, My_new_list, V_0, psi):
        ''' 
            ----
            Description: 
                Expands the wavefunction defined by the protocoll in Mx_new_list and converge again by imag time propagation
            ----

            ----
            Inputs:
                Mx_new_list (1-dimensional: (several points defined by user)): list of Mx values the grid should be sequentially increased
                My_new_list (1-dimensional: (several points defined by user)): list of My values the grid should be sequentially increased
                V_0 (scalar): interaction for this wavefunction
                psi (3-dimensional: (My,Mx,n)): wavefunction to expand
            ----

            ----
            Outputs:
                psi_new (3-dimensional: (My_new[end of list], Mx_new[end of list], n)): normalized wavefunction with bigger number of rotors
            ----
        '''

        for i in range(1,len(Mx_new_list)):
            if i == 1:
                psi_new = psi.copy()
            #params_conv = self.param_dict.copy()
        
            self.Mx = Mx_new_list[i-1]
            self.My = My_new_list[i-1]
    
            '''
                Here: manipulate wavefunction and add objects 
            '''
            self.param_dict['Mx_new'] = Mx_new_list[i]
            self.param_dict['My_new'] = My_new_list[i]
            psi_new = self.add_rotors_to_wavefunction(psi_new)

            # imag time propagation of expanded wavefunction
            tic = time.perf_counter() # start timer

            self.param_dict['Mx'] = Mx_new_list[i]
            self.param_dict['My'] = My_new_list[i]

            eom_object = eom.eom(params=self.param_dict)
            eom_object.V_0 = V_0  
            psi_new = eom_object.solve_for_fixed_params_imag_time_prop(psi_new)
        
            toc = time.perf_counter() # end timer
            print("\nExecution time = ", (toc-tic)/60, "min")

        return psi_new

class permute_rotors:
    ''' Class for moving the rotors in the different directions
        ----
        
        ----
        Inputs: 
            psi (3-dimensional: (My,Mx,n)): input psi
        ----

        ----
        Outputs:
            psi (3-dimensional: (My,Mx,n)): but here, one column or row was shifted
        ----

        ----
        Methods: 
            (Note for below: convention in array: [My,Mx,n]; [i,j] means picking the (i-th,j-th) rotor from the array)
            self.get_next_y_rotor(): "equivalent" to [(i+1)%self.My,j]
            self.get_prev_y_rotor(): "equivalent" to [i-1,j]
            self.get_next_x_rotor(): "equivalent" to [i,(j+1)%self.Mx]
            self.get_prev_x_rotor(): "equivalent" to [i,j-1]
        ----
    '''

    def __init__(self, psi):
        self.psi = psi

    def get_next_y_rotor(self):
        return np.roll(self.psi, 1, axis=0)
    
    def get_prev_y_rotor(self):
        return np.roll(self.psi, -1, axis=0)
    
    def get_next_x_rotor(self):
        return np.roll(self.psi, 1, axis=1)
    
    def get_prev_x_rotor(self):
        return np.roll(self.psi, -1, axis=1)