import numpy as np
from scipy.integrate import solve_ivp

import os, sys

path = os.path.dirname(__file__) 
sys.path.append(path)

# import user-defined classes
import class_energy as energy
import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

class eom:
    ''' Class for evaluation of the equations of motion for a rotor lattice ...

        ----
        Inputs:
            params: dictionary with all calculation parameters
        ----

        Important variables (mainly for ourput/debugging):
        
        ----
        Calculation parameters and class variables:
            n (int): length of angle grid
            Mx (int): number of rotor lattice in x direction
            My (int): number of rotor lattice in y direction

            tx (float): tunneling ratio in x direction
            ty (float): tunneling ratio in y direction
            V_0 (float): coupling strength of interaction term
            B (float): rotational constant
            qx (int): wavenumber of electron in x direction
            qy (int): wavenumber of electron in y direction

            dt (float): time step of one Runge-Kutta propagation
            time_steps (int): number of time steps in the real time propagation
            tol (float): convergence criterion for the ground state in the imaginary time propagation
        ----

        but most importantly:

        ----
        Methods:
            self.hpsi_lang_firsov(wavefunc. as three-dimensional numpy array): calculate H_psi
                                                        of the variational equation of motion
            self.rhs_lang_firsov_imag_time_prop(wavefunc. as three-dimensional numpy array):
            self.rhs_lang_firsov_real_time_prop(wavefunc. as three-dimensional numpy array):
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
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
        self.dt  = float(params['dt'])

    def hpsi_lang_firsov(self, psi_collection):
        '''
            Computes: H_psi of the variational equations of motion

            ----
            Inputs:
                psi_collection (3-dimensional: (My, Mx, n), dtype: complex): stores the rotor wavefunctions
            ----

            ----
            Variables: 
                TD_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for down jumping 
                TU_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for up jumping

                TR_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for right jumping
                TL_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for left jumping
            ----

            ----
            Outputs:
                H_psi (2-dimensional: (My*Mx,n)): H_psi of the variational equations of motion
            ----
        '''
        
        # object for manipulating wavefunctions
        wfn_manip = h_wavef.permute_rotors(psi_collection)

        #TL_arr = np.zeros((self.My,self.Mx), dtype=complex)
        #TR_arr = np.zeros((self.My,self.Mx), dtype=complex)
        #TU_arr = np.zeros((self.My,self.Mx), dtype=complex)
        #TD_arr = np.zeros((self.My,self.Mx), dtype=complex)

        # compute arrays for all the pairwise (column/row wise) overlaps
        #for k in range(self.My):
        #    for p in range(self.Mx):
        #        TD_arr[k,p] = np.sum(np.conjugate(psi_collection[k,p])*psi_collection[(k+1)%self.My,p])
        #        TU_arr[k,p] = np.sum(np.conjugate(psi_collection[k,p])*psi_collection[k-1,p])
        #    
        #        TR_arr[k,p] = np.sum(np.conjugate(psi_collection[k,p])*psi_collection[k,(p+1)%self.Mx])
        #        TL_arr[k,p] = np.sum(np.conjugate(psi_collection[k,p])*psi_collection[k,p-1])

        psi_collection_conj = np.conjugate(psi_collection)

        # compute transfer integrals 
        TD_arr = np.roll(np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_next_y_rotor(), dtype=complex), -1, axis=0)
        TU_arr = np.roll(np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_prev_y_rotor(), dtype=complex), 1, axis=0)
        TR_arr = np.roll(np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_next_x_rotor(), dtype=complex), -1, axis=1)
        TL_arr = np.roll(np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_prev_x_rotor(), dtype=complex), 1, axis=1)
        
        TD = np.prod(TD_arr)
        TU = np.prod(TU_arr)
        TR = np.prod(TR_arr)
        TL = np.prod(TL_arr)

        H_psi = np.zeros((self.My,self.Mx,self.n), dtype=complex) # create a matrix for every rotor 

        k2  = -np.append(np.arange(0,self.n/2+1),np.arange(-self.n/2+1,0))**2 # make second derivative matrix
        for i in range(self.My):
            for j in range(self.Mx):
                H_psi[i,j] = -self.B*np.fft.ifft(k2*np.fft.fft(psi_collection[i,j])) # kinetic energy term for every rotor

                TDr = TD / TD_arr[i, j]
                TUr = TU / TU_arr[i, j]
                TRr = TR / TR_arr[i, j]
                TLr = TL / TL_arr[i, j]

                # for every (i,j) rotor wave function we need to add the calculated transfer integrals
                H_psi[i,j] += - self.ty*( \
                    np.exp(-1j*(2*np.pi*self.qy/self.My))*TDr*psi_collection[(i+1)%self.My,j] \
                    + np.exp(+1j*(2*np.pi*self.qy/self.My))*TUr*psi_collection[i-1,j])
                H_psi[i,j] += - self.tx*( \
                    np.exp(-1j*(2*np.pi*self.qx/self.Mx))*TRr*psi_collection[i,(j+1)%self.Mx] \
                    + np.exp(+1j*(2*np.pi*self.qx/self.Mx))*TLr*psi_collection[i,j-1])

        # for the rotors adjacent to the electron add the potential terms
        H_psi[self.My-1,0]         += self.V_0*np.cos(self.x-0.25*np.pi)*psi_collection[self.My-1,0]
        H_psi[self.My-1,self.Mx-1] += self.V_0*np.cos(self.x-0.75*np.pi)*psi_collection[self.My-1,self.Mx-1]
        H_psi[0,0]                 += self.V_0*np.cos(self.x+0.25*np.pi)*psi_collection[0,0]
        H_psi[0,self.Mx-1]         += self.V_0*np.cos(self.x+0.75*np.pi)*psi_collection[0,self.Mx-1]

        H_psi = H_psi.reshape((self.M,self.n))
        return H_psi

    def rhs_lang_firsov_imag_time_prop(self, psi_collection):
        '''
            Computes: right-hand-side of the variational e.o.m. for imaginary time propagation

            ----
            Inputs:
                psi_collection (3-dimensional: (My, Mx, n), dtype: complex): stores the rotor wavefunctions
            ----

            ----
            Variables: 
                H_psi (2-dimensional: (My*Mx,n)): right-hand-side of the variational equations of motion
                lagrange_param (2-dimensional: (My*Mx,n)): lagrange parameter to ensure normalization
            ----

            ----
            Outputs:
                H_psi (1-dimensional: (My*Mx*n)): right-hand-side of the variational equations of motion for imag time evolution
            ----
        '''
        H_psi = self.hpsi_lang_firsov(psi_collection)

        psi_collection = psi_collection.reshape((self.M,self.n))
        lagrange_param = np.sum(np.conjugate(psi_collection)*H_psi,axis=1).reshape(self.M,1)*psi_collection
        
        H_psi = H_psi - lagrange_param
        H_psi = H_psi.reshape((self.M*self.n,))

        return H_psi

    def rhs_lang_firsov_real_time_prop(self, psi_collection):
        '''
            Computes: right-hand-side of the variational e.o.m. for real time propagation

            ----
            Inputs:
                psi_collection (3-dimensional: (My, Mx, n), dtype: complex): stores the rotor wavefunctions
            ----

            ----
            Variables: 
                H_psi (2-dimensional: (My*Mx,n)): right-hand-side of the variational equations of motion
            ----

            ----
            Outputs:
                H_psi (1-dimensional: (My*Mx*n)): right-hand-side of the variational equations of motion for imag time evolution
            ----
        '''
        H_psi = self.hpsi_lang_firsov(psi_collection)
        H_psi = H_psi.reshape((self.M*self.n,))

        return H_psi
    
    def create_integration_function_imag_time_prop(self): 
        '''
            Computes: lambda expression for imaginary time propagation

            ----
            Inputs:
                None
            ----

            ----
            Outputs:
                Lambda expression for scipy ivp solver
            ----
        '''
    
        return lambda t_, psi_ : -1.0*self.rhs_lang_firsov_imag_time_prop(psi_.reshape((self.My,self.Mx,self.n)))

    def create_integration_function_real_time_prop(self): 
        '''
            Computes: lambda expression for real time propagation

            ----
            Inputs:
                None
            ----

            ----
            Outputs:
                Lambda expression for scipy ivp solver
            ----
        '''

        return lambda t_, psi_ : 1j*self.rhs_lang_firsov_real_time_prop(psi_.reshape((self.My,self.Mx,self.n)))

    ## you need two functions that transform between the three dimensional and one-dimensional numpy representations in wavefunction manipulation things
    def solve_for_fixed_params_imag_time_prop(self, psi_init):
        '''
            Computes: finds ground state variational wave function for the defined parameters

            ----
            Inputs:
                psi_init (3-dimensional: (My,Mx,n)): initial wavefunction 
            ----

            ----
            Outputs:
                psi_out (3-dimensional: (My,Mx,n)): output wavefunction
            ----

            ----
            Logic:
                (1) evolve psi_init through time dt -> psi_iter
                (2) reshape psi_iter and normalize it
                (3) compute overlap with previous variational state, i.e. with psi_init
                (4) check whether epsilon criterion is converged
                (5) update psi_init and repeat (1) to (4)
            ----
        '''
        wfn_manip = h_wavef.wavefunc_operations(params=self.param_dict)
        func = self.create_integration_function_imag_time_prop() # lambda expression of right-hand-side of e.o.m

        iter = 0
        epsilon = 1 
        tol = self.param_dict['tol']
        while epsilon > tol:
            print('V_0 =', self.V_0, ', iter step = ' + str(iter+1))
        
            sol = solve_ivp(func, [0,self.dt], psi_init.copy(), method='RK45', rtol=1e-9, atol=1e-9) # method='RK45','DOP853'

            # norm function you could also use einsum and the transformation functions
            psi_iter = sol.y.T[-1]
            psi_iter = wfn_manip.normalize_wf(psi_iter, shape=(self.M,self.n)) #(1.0/np.sqrt(np.sum(np.abs(psi_iter)**2,axis=1))).reshape(self.M,1) * psi_iter # normalization for numerical errors

            #epsilon = 4 - np.sum(np.conjugate(psi_iter_before.reshape((M*n)))*psi_col) # indication of convergence
            #epsilon = 1 - np.max(np.sum(np.conjugate(psi_iter_before.reshape((M,n)))*psin, axis=1)) # indication of convergence
            epsilon = 1 - np.abs(np.min(np.sum(np.conjugate(psi_init.reshape((self.M,self.n)))*psi_iter, axis=1))) # indication of convergence
        
            print("epsilon =", epsilon, "\n")
            psi_init = psi_iter.reshape((self.M*self.n)) # update psi_init

            iter = iter + 1

        psi_out = psi_init
        return psi_out
    

    def solve_for_fixed_params_real_time_prop(self, psi_init, path_main):
        '''
            Computes: real time evolution of variational wave function for the defined parameters

            ----
            Inputs:
                psi_init (3-dimensional: (My,Mx,n)): initial wavefunction - typically uniform
                path_main (string): path to the folder of original .py file
            ----

            ----
            Outputs:
                None - everything is saved already here
            ----

            ----
            Logic:
                (1) evolve psi_curr through time dt 
                (2) compute overlap of psi_curr with psi_init
                (3) compute normalization
                (4) compute energy of psi_curr
                (5) repeat (1) to (4)
            ----
        '''

        # input object for storing the results
        in_object = h_in.green_function(params=self.param_dict)
        folder_name_g, file_name_green = in_object.result_folder_structure_real_time_prop(path_main) # get the folder structure for results
        folder_name_w, file_name_wavefunction = in_object.wavefunction_folder_structure_real_time_prop(path_main) # get the folder structure for wavefunctions

        wfn_manip = h_wavef.wavefunc_operations(params=self.param_dict)

        # energy objects
        energy_object = energy.energy(Mx=self.Mx, My=self.My, B=self.B, V_0=self.V_0, tx=self.tx, ty=self.ty,
                                    qx=self.qx, qy=self.qy, n=self.n, x=self.x, dt=self.dt, tol=self.tol) 
        overlap_object = energy.coupling_of_states(Mx=self.Mx, My=self.My, B=self.B, V_0=self.V_0, tx=self.tx, ty=self.ty,
                                    n=self.n, x=self.x, dt=self.dt, tol=self.tol) # needed for overlap calculations
        
        # lambda expression for right-hand-side of e.o.m
        func = self.create_integration_function_real_time_prop() 

        psi_curr = psi_init
        iter = 0 # time step variable
        max_iter = self.param_dict['time_steps']
        for iter in range(max_iter):
            print('V_0 =', self.V_0, ', time step = ' + str(iter+1), ' of', max_iter)

            # evolution in imaginary time # method='RK45','DOP853'
            sol = solve_ivp(func, [0,self.dt], psi_curr, method='RK45', rtol=1e-9, atol=1e-9) 
            psi_curr = sol.y.T[-1] # don't normalize result!?
            
            norm = np.sum(1./wfn_manip.normalization_factor_wf(psi_curr))/self.M
            green_function = overlap_object.calc_overlap(psi_curr, psi_init) 
            E = np.asarray(energy_object.calc_energy(psi_curr))

            print("Green   =", green_function)
            print("Energy  =", E)
            print("Norm    =", norm, "\n")

            # save the green function and energy values
            np.save(folder_name_w+file_name_wavefunction+str(iter), psi_curr.reshape(self.My,self.Mx,self.n)) # save wavefunction
            with open(folder_name_g+file_name_green, 'a') as green_f_file:
                write_string = str(green_function)+' '+str(E[0])+' '+str(E[1])+' '+str(E[2])+' '+str(E[3])+'\n'
                green_f_file.write(write_string)

        return 
