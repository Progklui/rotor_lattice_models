import numpy as np
from scipy.integrate import solve_ivp

import os, sys

'''
    - Goal of the code: solve the equations of motion and provide respective outputs
    - Called: either by real_time_propagation.py or imag_time_propagation.py

    - Philosophy: 
        - one function for providing the right hand side (rhs) of the equations of motion (e.o.m)
        - one function for creating a lambda expression of the rhs
        - two functions for performing the real or imaginary time evolutions
'''

# import user-defined classes
import class_energy as energy
import class_handle_input as h_in
import class_handle_wavefunctions as h_wavef

path = os.path.dirname(__file__) 
sys.path.append(path)

class eom:
    ''' Class for evaluation of the equations of motion for a rotor lattice ...
        Here you should write an overview of inputs like this:

        inputs
        ------
        Mx (float): length of rotor lattice in x direction
        ...

        Important variables (mainly for ourput/debugging)

        variables
        ---------
        self.*** (x-dimensional numpy array) description

        but most importantly:

        methods
        -------
        self.rhs_lang_firsov(three-dimensional numpy array) calculate the right hand side of the
                                                    variational equation of motion
    '''    
    def __init__(self, Mx, My, B, V_0, tx, ty, qx, qy, n, dt, tol):
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.V_0 = V_0
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.x   = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid
        self.dt  = dt
        self.tol = tol
        self.real_or_image_time_unit = 1j
        self.real_or_imag_time = 1. # real time: = 0., imag time: = 1., parameter for including lagrange parameter

    ## split this into two functions
    # 1 function is hpsi_lang_firsov
    # 2 function is rhs_lang_firsov here you call hpsi_lang_firsov and evaluate the Langrange multipliers
    def rhs_lang_firsov(self, psi_collection):
        '''
        Again here you should write explicity the inputs and outputs of this function and also in more detail what each dimension of each matrix is
    

            This is the right-hand-side of the e.o.m. for real and imaginary time propagation:
        
            Switching between real and imag time:
                - real-time propagation: real_or_imag_time = 0., the right hand side of the equation doesn't have the lagrange multiplier that constrains the wavefunction
                - image-time propagation: real_or_imag_time = 1., we need the lagrange multipliers
        '''

        TL_arr = np.zeros((self.My,self.Mx), dtype=complex)
        TR_arr = np.zeros((self.My,self.Mx), dtype=complex)
        TU_arr = np.zeros((self.My,self.Mx), dtype=complex)
        TD_arr = np.zeros((self.My,self.Mx), dtype=complex)

        # compute arrays for all the pairwise (column/row wise) overlaps
        for k in range(self.My):
            for p in range(self.Mx):
                TD_arr[k,p] = np.sum(np.conjugate(psi_collection[k,p])*psi_collection[(k+1)%self.My,p])
                TU_arr[k,p] = np.sum(np.conjugate(psi_collection[k,p])*psi_collection[k-1,p])
            
                TR_arr[k,p] = np.sum(np.conjugate(psi_collection[k,p])*psi_collection[k,(p+1)%self.Mx])
                TL_arr[k,p] = np.sum(np.conjugate(psi_collection[k,p])*psi_collection[k,p-1])

        # psi_collection_conj = np.conjugate(psi_collection)
        # TD_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manipulation.get_next_y(psi_collection))
        # TU_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manipulation.get_prev_y(psi_collection))
        # TR_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manipulation.get_next_x(psi_collection))
        # TL_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manipulation.get_prev_x(psi_collection))
        
        H_psi = np.zeros((self.My,self.Mx,self.n), dtype=complex) # create a matrix for every rotor 

        TD = np.prod(TD_arr)
        TU = np.prod(TU_arr)
        TR = np.prod(TR_arr)
        TL = np.prod(TL_arr)

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
    
        M = int(self.My*self.Mx)

        # Here we make the switch from real time to imaginary time evolution
        H_psi = H_psi.reshape((M,self.n)) - self.real_or_imag_time*np.sum(np.conjugate(psi_collection.reshape((M,self.n)))\
            *H_psi.reshape((M,self.n)),axis=1).reshape(M,1)*psi_collection.reshape((M,self.n)) 
        return H_psi.reshape((M*self.n,))

    # create lambda expression - Attention: prefactor determines time evolution!
    def create_integration_function(self): 
        return lambda t_, psi_ : -1.0*self.real_or_image_time_unit*self.rhs_lang_firsov(psi_.reshape((self.My,self.Mx,self.n)))

    '''
        function to solve for the imaginary time propagation
    '''
    ## you need two functions that transform between the three dimensional and one-dimensional numpy representations in wavefunction manipulation things
    def solve_for_fixed_coupling_imag_time_prop(self, psi_col):
        M = int(self.My*self.Mx) # compute total number of rotors - not 'trivial' for Mx \neq My

        # definitions for the time evolution
        self.real_or_image_time_unit = 1. # the sign as defined in self.create_integration_function is okay this way - don't change something
        self.real_or_imag_time = 1. # we want imag time propagation
        func = self.create_integration_function() # get lambda expression of right-hand-side of e.o.m

        psi_iter_before = psi_col # updated to calculate the overlap between consecutive steps

        iter = 0
        epsilon = 1 # initialize epsilon
        # code logic: take evolve psi_col through dt, then check whether it has converged, update psi_col and then evolve again through time dt
        while epsilon > self.tol:
            print('V_0 =', self.V_0, ', iter step = ' + str(iter+1))
        
            sol = solve_ivp(func, [0,self.dt], psi_col.copy(), method='RK45', rtol=1e-9, atol=1e-9) # evolution in imaginary time # method='RK45','DOP853'
        
            psin = sol.y.T[-1]
            # norm function you could also use einsum and the transformation functions
            psin = (1.0/np.sqrt(np.sum(np.abs(psin.reshape((M,self.n)))**2,axis=1))).reshape(M,1) * psin.reshape((M,self.n)) # normalization for numerical errors
        
            psi_col = psin.reshape((M*self.n,)).copy()

            #epsilon = 4 - np.sum(np.conjugate(psi_iter_before.reshape((M*n)))*psi_col) # indication of convergence
            #epsilon = 1 - np.max(np.sum(np.conjugate(psi_iter_before.reshape((M,n)))*psin, axis=1)) # indication of convergence
            epsilon = 1 - np.abs(np.min(np.sum(np.conjugate(psi_iter_before.reshape((M,self.n)))*psin, axis=1))) # indication of convergence
        
            print("epsilon =", epsilon, "\n")
            psi_iter_before = psin.copy() # update to calculate next overlap needed for the evaluation of epsilon

            iter = iter + 1 # to know in which step one is

        return psi_col.copy() # return (M*n) array containing the wavefunction for the specified parameters
    
    '''
        function to solve for the real time propagation
    '''
    def solve_for_fixed_coupling_real_time_prop(self, path_main, time_steps):

        # input object for storing the results
        in_object = h_in.green_function(Mx=self.Mx, My=self.My, B=self.B, V_0=self.V_0, tx=self.tx, ty=self.ty, 
                                        qx=self.qx, qy=self.qy, n=self.n, dt=self.dt, time_steps=time_steps)
        folder_name_g, file_name_green = in_object.result_folder_structure_real_time_prop(path_main) # get the folder structure for results
        folder_name_w, file_name_wavefunction = in_object.wavefunction_folder_structure_real_time_prop(path_main) # get the folder structure for wavefunctions

        # energy objects
        energy_object = energy.energy(Mx=self.Mx, My=self.My, B=self.B, V_0=self.V_0, tx=self.tx, ty=self.ty,
                                    qx=self.qx, qy=self.qy, n=self.n, x=self.x, dt=self.dt, tol=self.tol) 
        overlap_object = energy.coupling_of_states(Mx=self.Mx, My=self.My, B=self.B, V_0=self.V_0, tx=self.tx, ty=self.ty,
                                    n=self.n, x=self.x, dt=self.dt, tol=self.tol) # needed for overlap calculations

        # definitions for the time evolution
        self.real_or_image_time_unit = -1j # now we want to have the imaginary part in the time evolution!
        self.real_or_imag_time = 0. # we want imag time propagation
        func = self.create_integration_function() # get lambda expression of right-hand-side of e.o.m

        # create uniform initial wavefunction
        wavefunc_object = h_wavef.wavefunctions(Mx=self.Mx, My=self.My, B=self.B, V_0=self.V_0, tx=self.tx, ty=self.ty, 
                                                qx=self.qx, qy=self.qy, n=self.n, x=self.x, dt=self.dt, tol=self.tol)
        psi_0 = wavefunc_object.create_init_wavefunction('uniform') # store the initial configuration (uniform) to compute the overlap at every time step
        psi_curr = psi_0.copy()

        '''
            Code logic: 
                (1) evolve psi_col until dt
                (2) compute green function, i.e. overlap with uniform psi
                (3) evolve the updated psi_col further 
                - repeat (1) to (3)
        ''' 
        iter = 0 # time step variable
        for iter in range(time_steps):
            print('V_0 =', self.V_0, ', time step = ' + str(iter+1))

            # evolution in imaginary time # method='RK45','DOP853'
            sol = solve_ivp(func, [0,self.dt], psi_curr, method='RK45', rtol=1e-9, atol=1e-9) 
        
            psi_curr = sol.y.T[-1] # don't normalize result
            #psin = (1.0/np.sqrt(np.sum(np.abs(psin.reshape((M,self.n)))**2,axis=1))).reshape(M,1) * psin.reshape((M,self.n)) # normalization for numerical errors
        
            green_function = overlap_object.calc_overlap(psi_curr, psi_0) 
            E = np.asarray(energy_object.calc_energy(psi_curr)) # compute energy

            print("Green   =", green_function)
            print("Energy  =", E, "\n")

            # save the green function and energy values
            np.save(folder_name_w+file_name_wavefunction+str(iter), psi_curr.reshape(self.My,self.Mx,self.n)) # save wavefunction
            with open(folder_name_g+file_name_green, 'a') as green_f_file:
                write_string = str(green_function)+' '+str(E[0])+' '+str(E[1])+' '+str(E[2])+' '+str(E[3])+'\n'
                green_f_file.write(write_string)

        return # no return, as everything is already saved here
