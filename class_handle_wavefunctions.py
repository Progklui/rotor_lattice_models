import numpy as np
import scipy 

import os, sys
path = os.path.dirname(__file__) 
sys.path.append(path)

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
            Todo: make external functional that creates the angle grid - such that this can be changed externally somehow!
        '''
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid

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
                    - phase == 'small_polaron': analytic Mathieu function
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
        # psi_init is expected in the format Mx*My*n
        psi_init = self.n**(-0.5)*np.ones((int(self.Mx*self.My)*self.n),dtype=complex)
        psi_init = psi_init.reshape((self.My,self.Mx,self.n))

        if phase == 'uniform':
            psi_init = self.n**(-0.5)*np.ones((int(self.Mx*self.My)*self.n),dtype=complex)
        
        # initialization for the ferroelectric domain, wall oriented vertically
        elif phase == 'ferro_domain_vertical_wall': 
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
                
        # initialization for the ferroelectric domain, wall oriented horizontally   
        elif phase == 'ferro_domain_horizontal_wall': 
            for j in range(self.Mx):
                # top row
                psi_init[self.My-1,j] = np.sin(self.x) 
                psi_init[self.My-1,j][0:int(self.n/2)] = 0.01 

                # bottom row
                psi_init[0,j] = np.sin(self.x) 
                psi_init[0,j][int(self.n/2):self.n] = 0.01 
        
        # random initialization
        elif phase == 'random': 
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

        # analytic small polaron wavefunctions
        elif phase == 'small_polaron': 
            # bottom left
            y, yp = scipy.special.mathieu_cem(0, 2*self.V_0/self.B, (self.x+3*np.pi/4)/2*180/np.pi)
            psi_init[0,self.Mx-1] = y/np.sqrt(np.sum(y*y))
        
            # bottom right
            y, yp = scipy.special.mathieu_cem(0, 2*self.V_0/self.B, (self.x+np.pi/4)/2*180/np.pi)
            psi_init[0,0] = y/np.sqrt(np.sum(y*y))

            # top left
            y, yp = scipy.special.mathieu_cem(0, 2*self.V_0/self.B, (self.x-3*np.pi/4)/2*180/np.pi)
            psi_init[self.My-1,self.Mx-1] = y/np.sqrt(np.sum(y*y))

            # top right
            y, yp = scipy.special.mathieu_cem(0, 2*self.V_0/self.B, (self.x-np.pi/4)/2*180/np.pi)
            psi_init[self.My-1,0] = y/np.sqrt(np.sum(y*y))

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
        self.Mx  = int(params['Mx'])
        self.My  = int(params['My'])
        self.M   = int(params['Mx']*params['My'])
        self.n   = int(params['n'])

    def normalization_factor_wf(self, psi):
        '''
            ----
            Inputs:
                psi (shape doesn't matter: max. 3-dimensional):
            ----

            ----
            Outputs:
                normalization factor (shape: (Mx*My,1)): gives the normalization factor for every rotor
            ----
        '''
        psi = psi.reshape((int(self.Mx*self.My),self.n))

        norm = np.sqrt(np.sum(np.abs(psi)**2,axis=1)).reshape(self.M,1)
        normalization_factor = 1.0/norm #.reshape(int(self.Mx*self.My))
        return normalization_factor
    
    def normalize_wf(self, psi, shape):
        '''
            ----
            Inputs:
                psi (shape doesn't matter: max. 3-dimensional): wavefunction to normalize
            ----

            ----
            Outputs:
                psi (shape as specified by input shape=(,,)): normalized wavefunction
            ----
        '''
        normalization_factor = self.normalization_factor_wf(psi)
        psi = normalization_factor*psi.reshape((self.My*self.Mx,self.n)) #.reshape(shape)
        return psi
    
    def reshape_one_dim(self, psi):
        return psi.reshape((self.My*self.Mx*self.n))
    
    def reshape_two_dim(self, psi):
        return psi.reshape((self.My*self.Mx,self.n))
    
    def reshape_three_dim(self, psi):
        return psi.reshape((self.My,self.Mx,self.n))
    '''
        TODo: here the overlap function, which is referenced from outside, e.g. from the equations of motion and energy object
    '''
    
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