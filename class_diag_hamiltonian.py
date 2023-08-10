import numpy as np
import scipy 

import os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

import class_equations_of_motion as eom 
import class_handle_wavefunctions as h_wavef
import class_energy as energy

class diagonalization:
    ''' Class for diagonalizing the effective hamiltonian

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
        ----

        but most importantly:

        ----
        Methods:
            
        ----
    '''

    def __init__(self, params):
        self.param_dict = params
        self.Mx  = int(params['Mx'])
        self.My  = int(params['My'])
        self.M   = int(params['Mx']*params['My'])
        self.B   = float(params['B'])
        self.V_0 = 0 if isinstance(params['V_0'], list) == True else float(params['V_0']) # for safety, update, set outside!
        self.tx  = float(params['tx'])
        self.ty  = float(params['ty'])
        self.qx  = int(params['qx'])
        self.qy  = int(params['qy'])
        self.exc_states = int(params['excitation_no'])
        self.n   = int(params['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
        self.sigma_gaussian = 0 # set from outside - width of the gaussian

    def first_derivative_mat(self):
        '''
            Computes: first derivative operator on fourier grid
        '''

        '''
        first derivative matrix
        '''
        k1 = 1j*np.fft.ifftshift(np.arange(-self.n/2,self.n/2))
        K1 = np.diag(k1)

        four_matrix = scipy.linalg.dft(self.n)

        '''
        first deriv. operator
        '''
        first_derivative = (np.matmul(four_matrix.conj().transpose(), np.matmul(K1, four_matrix)))/self.n 

        return first_derivative

    def second_derivative_mat(self):
        '''
            Computes: second derivative operator on fourier grid
        '''
 
        '''
        second derivative matrix
        '''
        k2 = -np.append(np.arange(0,self.n/2+1),np.arange(-self.n/2+1,0))**2 
        K2 = np.diag(k2)

        four_matrix = scipy.linalg.dft(self.n)
        
        '''
        second deriv. operator
        '''
        sec_derivative = (np.matmul(four_matrix.conj().transpose(), np.matmul(K2, four_matrix)))/self.n 

        return sec_derivative

    def get_state_i(self, e_ket_arr, i):
        y_theory = e_ket_arr[:,i]
        norm = np.sqrt(np.abs(np.sum(y_theory*y_theory)))
        y_theory = y_theory / norm 

        return y_theory

    def projector(self, psi, i1, j1,  i2, j2):
        '''
            Computes: outer product of rotor (i1,j1) with rotor (i2,j2)
        '''
        psi_n = psi.copy() 

        psi_n_conjugate = np.conjugate(psi_n)

        return psi_n[i1,j1,:][:, np.newaxis] * psi_n_conjugate[i2,j2,:][np.newaxis, :]
    
    def lagrange_multiplier(self, psi, i, j, H_psi):
        '''
            Computes: Lagrange Multiplier for wavefunction psi

            ----
            Inputs:
                psi (1-dimensional: (n), dtype=complex): wavefunction for a selected rotor
                H_psi (2-dimensional: (n,n), dtype=complex): eff. Hamiltonian projected out of rotor psi
            ----

            ----
            Outputs:
                lag_mul (2-dimensional: (n,n), dtype=complex): lagrange multiplier matrix 
            ----
        '''
        psi_conj = np.conjugate(psi)

        eom_object = eom.eom(params=self.param_dict)
        eom_object.V_0 = self.V_0 

        H_psi = eom_object.hpsi_lang_firsov(psi)
        lag_mul = np.einsum('ijk,ijk->ij', psi_conj, H_psi)

        lag_mul = np.diag(lag_mul[i,j]*np.ones(self.n, dtype=complex))
        
        #lag_mul = np.diag(np.einsum('n,nn->n', psi_conj, H_psi))

        return lag_mul
    
    def eff_individual_projector_terms(self, psi, i, j):
        psi = psi.copy()

        '''
        objects to store the excitation wavefunctions and e-vals
        '''
        energy_object = energy.energy(params=self.param_dict)

        '''
        transfer integrals and matrices
        '''
        TD_arr, TU_arr, TR_arr, TL_arr = energy_object.transfer_matrices(psi, psi)
                
        '''
        transfer terms
        '''
        TDr = self.transfer_integrals(TD_arr, i, j, i-1, j) # TD / (TD_arr[i, j]*TD_arr[i-1, j])
        TUr = self.transfer_integrals(TU_arr, i, j, (i+1)%self.My, j) # TU / (TU_arr[i, j]*TU_arr[(i+1)%self.My, j])
        TRr = self.transfer_integrals(TR_arr, i, j, i, j-1) # TR / (TR_arr[i, j]*TR_arr[i, j-1])
        TLr = self.transfer_integrals(TL_arr, i, j, i, (j+1)%self.Mx) # TL / (TL_arr[i, j]*TL_arr[i, (j+1)%self.Mx])

        TDr_projector = np.exp(-1j*(2*np.pi*self.qy/self.My))*TDr*self.projector(psi, (i+1)%self.My, j, i-1, j)
        TUr_projector = np.exp(+1j*(2*np.pi*self.qy/self.My))*TUr*self.projector(psi, i-1, j, (i+1)%self.My, j)
        TRr_projector = np.exp(-1j*(2*np.pi*self.qx/self.Mx))*TRr*self.projector(psi, i, (j+1)%self.Mx, i, j-1)
        TLr_projector = np.exp(+1j*(2*np.pi*self.qx/self.Mx))*TLr*self.projector(psi, i, j-1, i, (j+1)%self.Mx)

        return TDr_projector, TUr_projector, TRr_projector, TLr_projector
    
    def single_rotor_eff_fdv_bias_potential(self, psi, i, j):
        '''
            Computes: effective biased potential for the i,j-th domain wall rotor

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n), dtype=complex): wavefunction on which the eff. Hamiltonian depends
                i (scalar, int): rotor index along My
                j (scla)
            ----

            ----
            Outputs:
                energy_exc_states (3-dimensional: (My,Mx,exc_states)): for every rotor and exc. number the resp. energy
                psi_exc_states (4-dimensional: (My,Mx,exc_states,n)): for every rotor and exc. number the respective wavefunction
                                                                    in the last dimension
            ----
        '''
        psi = psi.copy()

        '''
        objects to store the excitation wavefunctions and e-vals
        '''
        energy_object = energy.energy(params=self.param_dict)

        '''
        gaussian centered around pi and normalize
        '''
        dx = self.x[1]-self.x[0]
        w = 2*np.pi*np.fft.fftfreq(self.x.size, dx)
        
        gauss_pi = np.exp(-(self.x-np.pi)**2/(2*self.sigma_gaussian**2))
        norm = np.sum(np.abs(np.exp(-(self.x-np.pi)**2/(2*self.sigma_gaussian**2)))**2)
        gauss_pi = 1/(norm**0.5)*gauss_pi

        '''
        shift gaussian appropriately
        '''
        gauss_zero = np.fft.ifft(np.exp(1.0j*np.angle(self.x-2*np.pi)*w)*np.fft.fft(gauss_pi))

        '''
        transfer integrals and matrices
        '''
        TD_arr, TU_arr, TR_arr, TL_arr = energy_object.transfer_matrices(psi, psi)
                
        '''
        transfer terms
        '''
        TDr = self.transfer_integrals(TD_arr, i, j, i-1, j) # TD / (TD_arr[i, j]*TD_arr[i-1, j])
        TUr = self.transfer_integrals(TU_arr, i, j, (i+1)%self.My, j) # TU / (TU_arr[i, j]*TU_arr[(i+1)%self.My, j])
        TRr = self.transfer_integrals(TR_arr, i, j, i, j-1) # TR / (TR_arr[i, j]*TR_arr[i, j-1])
        TLr = self.transfer_integrals(TL_arr, i, j, i, (j+1)%self.Mx) # TL / (TL_arr[i, j]*TL_arr[i, (j+1)%self.Mx])

        V_psi = -self.ty*(np.exp(-1j*(2*np.pi*self.qy/self.My))*TDr*self.projector(psi, (i+1)%self.My, j, i-1, j)\
                    + np.exp(+1j*(2*np.pi*self.qy/self.My))*TUr*self.projector(psi, i-1, j, (i+1)%self.My, j))
        V_psi -= self.tx*(np.exp(-1j*(2*np.pi*self.qx/self.Mx))*TRr*self.projector(psi, i, (j+1)%self.Mx, i, j-1) \
                    + np.exp(+1j*(2*np.pi*self.qx/self.Mx))*TLr*self.projector(psi, i, j-1, i, (j+1)%self.Mx))

        if j == 0:
            V_psi = V_psi@np.diag(gauss_pi)
        if j == self.Mx-1:
            V_psi = V_psi@np.diag(gauss_zero)

        '''
        interaction terms for right rotors
        '''
        if i == (self.My-1) and j == 0: 
            V_psi += np.diag(self.V_0*np.cos(self.x-0.25*np.pi))*gauss_pi
        elif i == (self.My-1) and j == (self.Mx-1): 
            V_psi += np.diag(self.V_0*np.cos(self.x-0.75*np.pi))*gauss_zero
        elif i == 0 and j == 0: 
            V_psi += np.diag(self.V_0*np.cos(self.x+0.25*np.pi))*gauss_pi
        elif i == 0 and j == (self.Mx-1): 
            V_psi += np.diag(self.V_0*np.cos(self.x+0.75*np.pi))*gauss_zero
                
        '''
        Lagrange Multiplier
        '''
        lag_mul = self.lagrange_multiplier(psi, i, j, V_psi)
        #V_psi -= lag_mul

        return V_psi

    def single_rotor_eff_potential(self, psi, i, j):
        '''
            Computes: effective Potential for the i,j-th rotor

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n), dtype=complex): wavefunction on which the eff. Hamiltonian depends
                i (scalar, int): rotor index along My
                j (scla)
            ----

            ----
            Outputs:
                energy_exc_states (3-dimensional: (My,Mx,exc_states)): for every rotor and exc. number the resp. energy
                psi_exc_states (4-dimensional: (My,Mx,exc_states,n)): for every rotor and exc. number the respective wavefunction
                                                                    in the last dimension
            ----
        '''
        psi = psi.copy()

        '''
        objects to store the excitation wavefunctions and e-vals
        '''
        energy_object = energy.energy(params=self.param_dict)

        '''
        transfer integrals and matrices
        '''
        TD_arr, TU_arr, TR_arr, TL_arr = energy_object.transfer_matrices(psi, psi)
                
        '''
        transfer terms
        '''
        TDr = self.transfer_integrals(TD_arr, i, j, i-1, j) # TD / (TD_arr[i, j]*TD_arr[i-1, j])
        TUr = self.transfer_integrals(TU_arr, i, j, (i+1)%self.My, j) # TU / (TU_arr[i, j]*TU_arr[(i+1)%self.My, j])
        TRr = self.transfer_integrals(TR_arr, i, j, i, j-1) # TR / (TR_arr[i, j]*TR_arr[i, j-1])
        TLr = self.transfer_integrals(TL_arr, i, j, i, (j+1)%self.Mx) # TL / (TL_arr[i, j]*TL_arr[i, (j+1)%self.Mx])

        V_psi = -self.ty*(np.exp(-1j*(2*np.pi*self.qy/self.My))*TDr*self.projector(psi, (i+1)%self.My, j, i-1, j)\
                    + np.exp(+1j*(2*np.pi*self.qy/self.My))*TUr*self.projector(psi, i-1, j, (i+1)%self.My, j))
        V_psi -= self.tx*(np.exp(-1j*(2*np.pi*self.qx/self.Mx))*TRr*self.projector(psi, i, (j+1)%self.Mx, i, j-1) \
                    + np.exp(+1j*(2*np.pi*self.qx/self.Mx))*TLr*self.projector(psi, i, j-1, i, (j+1)%self.Mx))

        '''
        interaction terms for right rotors
        '''
        if i == (self.My-1) and j == 0: 
            V_psi += np.diag(self.V_0*np.cos(self.x-0.25*np.pi))
        elif i == (self.My-1) and j == (self.Mx-1): 
            V_psi += np.diag(self.V_0*np.cos(self.x-0.75*np.pi))
        elif i == 0 and j == 0: 
            V_psi += np.diag(self.V_0*np.cos(self.x+0.25*np.pi))
        elif i == 0 and j == (self.Mx-1): 
            V_psi += np.diag(self.V_0*np.cos(self.x+0.75*np.pi))
                
        '''
        Lagrange Multiplier
        '''
        lag_mul = self.lagrange_multiplier(psi, i, j, V_psi)
        #V_psi -= lag_mul

        return V_psi
    
    def transfer_integrals(self, T_arr, i1, j1, i2, j2):
        T_arr_new = T_arr.copy()

        T_arr_new[i1,j1] = 1.0+0j
        T_arr_new[i2,j2] = 1.0+0j

        T = np.prod(T_arr_new)
        return T

    def diag_h_eff(self, psi):
        '''
            Computes: effective Hamiltonian and diagonalizes it for every rotor

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n), dtype=complex): wavefunction on which the eff. Hamiltonian depends
            ----

            ----
            Outputs:
                energy_exc_states (3-dimensional: (My,Mx,exc_states)): for every rotor and exc. number the resp. energy
                psi_exc_states (4-dimensional: (My,Mx,exc_states,n)): for every rotor and exc. number the respective wavefunction
                                                                    in the last dimension
            ----
        '''
        psi = psi.copy()

        '''
        objects to store the excitation wavefunctions and e-vals
        '''
        psi_exc_states = np.zeros((self.My,self.Mx,self.exc_states,self.n), dtype=complex) 
        energy_exc_states = np.zeros((self.My,self.Mx,self.exc_states), dtype=complex)

        sec_derivative = self.second_derivative_mat() 

        '''
        1. Compute H_psi for every rotor
        2. Diagonalize H_psi -> wave function and spectrum
        '''
        for i in range(self.My):
            for j in range(self.Mx):
                ''' 
                rotor kinetic term
                '''
                H_psi = -self.B*sec_derivative 
            
                V_psi = self.single_rotor_eff_potential(psi, i, j)

                H_psi += V_psi 

                '''
                Lagrange Multiplier
                '''
                lag_mul = self.lagrange_multiplier(psi, i, j, H_psi)
                #H_psi -= lag_mul

                '''
                diagonalization of h_eff
                '''
                eigen_values, eigen_vector = np.linalg.eigh(H_psi)
                order = np.argsort(eigen_values)
                eigen_vector = eigen_vector[:,order]
                eigen_values = eigen_values[order]

                for n in range(self.exc_states):
                    psi_exc_n = self.get_state_i(eigen_vector, n)

                    psi_exc_states[i,j,n] = psi_exc_n
                    energy_exc_states[i,j,n] = eigen_values[n]

        return energy_exc_states, psi_exc_states

    def apply_fdv_bias_coefficients(self, psi):
        '''
            Computes: applies vertical ferro-domain bias coefficients 

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n), dtype=complex): wavefunction
            ----

            ----
            Outputs:
                psi (3-dimensional: (My,Mx,n), dtype=complex): wavefunction with multiplied bias coefficients for the respective rotors
            ----
        '''
        psi_n = psi.copy()
        
        '''
        gaussian centered around pi and normalize
        '''
        dx = self.x[1]-self.x[0]
        w = 2*np.pi*np.fft.fftfreq(self.x.size, dx)

        gauss_pi = np.exp(-(self.x-np.pi)**2/(2*self.sigma_gaussian**2))
        norm = np.sum(np.abs(np.exp(-(self.x-np.pi)**2/(2*self.sigma_gaussian**2)))**2)
        gauss_pi = 1/(norm**0.5)*gauss_pi

        '''
        shift gaussian appropriately
        '''
        gauss_zero = np.fft.ifft(np.exp(1.0j*np.angle(self.x-2*np.pi)*w)*np.fft.fft(gauss_pi))

        '''
        apply the respective coefficients for every rotor in the fdv domain wall and normalize individual rotors
        '''
        for i in range(self.My):
            norm1 = np.sum(np.abs(gauss_pi*psi_n[i,0])**2)
            psi_n[i,0] = 1/(norm1**0.5)*gauss_pi*psi_n[i,0]

            norm2 = np.sum(np.abs(gauss_zero*psi_n[i,self.Mx-1])**2)
            psi_n[i,self.Mx-1] = 1/(norm2**0.5)*gauss_zero*psi_n[i,self.Mx-1]
        
        return psi_n

    def diag_h_eff_bias_fdv(self, psi):
        '''
            Computes: effective Hamiltonian with bias in the wavefunction and diagonalizes it for every rotor

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n), dtype=complex): wavefunction on which the eff. Hamiltonian depends
            ----

            ----
            Outputs:
                energy_exc_states (3-dimensional: (My,Mx,exc_states)): for every rotor and exc. number the resp. energy
                psi_exc_states (4-dimensional: (My,Mx,exc_states,n)): for every rotor and exc. number the respective wavefunction
                                                                    in the last dimension
            ----
        '''
        psi = psi.copy()
        psi = self.apply_fdv_bias_coefficients(psi) 

        '''
        objects to store the excitation wavefunctions and e-vals
        '''
        psi_exc_states = np.zeros((self.My,self.Mx,self.exc_states,self.n), dtype=complex) 
        energy_exc_states = np.zeros((self.My,self.Mx,self.exc_states), dtype=complex)

        '''
        1st and 2nd derivative matrices, both size = (n,n)
        '''
        first_derivative = self.first_derivative_mat()
        sec_derivative = self.second_derivative_mat() 

        '''
        1. Compute H_psi for every rotor
        2. Diagonalize H_psi -> wave function and spectrum
        '''
        for i in range(self.My):
            for j in range(self.Mx):
                ''' 
                rotor kinetic term
                '''
                H_psi = -self.B*sec_derivative 
            
                V_psi = self.single_rotor_eff_fdv_bias_potential(psi, i, j)

                H_psi += V_psi 

                '''
                Lagrange Multiplier
                '''
                lag_mul = self.lagrange_multiplier(psi, i, j, H_psi)
                #H_psi -= lag_mul

                '''
                diagonalization of h_eff
                '''
                eigen_values, eigen_vector = np.linalg.eigh(H_psi)
                order = np.argsort(eigen_values)
                eigen_vector = eigen_vector[:,order]
                eigen_values = eigen_values[order]

                for n in range(self.exc_states):
                    psi_exc_n = self.get_state_i(eigen_vector, n)

                    psi_exc_states[i,j,n] = psi_exc_n
                    energy_exc_states[i,j,n] = eigen_values[n]

        return energy_exc_states, psi_exc_states

class multi_ref_ci:
    ''' Class to prepare the multi ref ci states

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

            exc_states (int): exc_states-1 gives the number of excitated states to consider on every rotor
        ----

        but most importantly:

        ----
        Methods:
            
        ----
    '''

    def __init__(self, params):
        self.param_dict = params
        self.Mx  = int(params['Mx'])
        self.My  = int(params['My'])
        self.M   = int(params['Mx']*params['My'])
        self.B   = float(params['B'])
        self.V_0 = 0 if isinstance(params['V_0'], list) == True else float(params['V_0']) # for safety, update, set outside!
        self.tx  = float(params['tx'])
        self.ty  = float(params['ty'])
        self.qx  = int(params['qx'])
        self.qy  = int(params['qy'])
        self.exc_states = int(params['excitation_no'])
        self.n   = int(params['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
        self.set_phase_bool = True # bool to control whether to set phase or not

    def set_phase(self, psi):
        '''
            Computes: phase of the wavefunction psi

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n), dtype=complex)
            ----

            ----
            Outputs:
                psi: though phase set to 1 if self.set_phase_bool == True (global class flag, set outside!)
            ----
        '''

        psi_man = psi.copy() 

        if self.set_phase_bool == True:
            sign = np.sum((1/self.n**0.5)*psi_man)
            sign = sign/np.abs(sign) 

            psi_man = np.sign(sign.real)*psi_man #np.conjugate(sign)*psi_man

        return psi_man

    def get_indices_for_rotor_ij(self, i, j, index_list_rotor_1, index_list_rotor_2):
        '''
            Computes: index list with which the i,j-th rotor is interacting

            ----
            Inputs:
                i (scalar, int): rotor index along My axis
                j (scalar, int): rotor index along Mx axis
                index_list_rotor_1 (list, dtype=int): a succession of (i,j)-pairs, in the end contains the 
                    first rotors that interact with the ones of index_list_rotor_2
                index_list_rotor_2 (list, dtype=int): a successon of (k,p)-pairs, with which the rotor 1 interacts 
            ----

            ----
            Outputs:
                index_list_rotor_1
                index_list_rotor_2
            ----
        '''
        for p in range(j+1,self.Mx):
            index_arr_rotor_1 = np.array([i,j])
            index_arr_rotor_2 = np.array([i,p])

            index_list_rotor_1.append(index_arr_rotor_1)
            index_list_rotor_2.append(index_arr_rotor_2)

        for k in range(i+1,self.My):
            for p in range(self.Mx):
                index_arr_rotor_1 = np.array([i,j])
                index_arr_rotor_2 = np.array([k,p])

                index_list_rotor_1.append(index_arr_rotor_1)
                index_list_rotor_2.append(index_arr_rotor_2)

        return index_list_rotor_1, index_list_rotor_2

    def get_double_rotor_excitation_list(self):
        '''
            Computes: computes index lists of rotor pairs that interact with one another

            ----
            Inputs:
                None
            ----

            ----
            Outputs:
                index_list_rotor_1 (list with M*(M-1)/2 entries): rotor 1 indices that "interact" with rotor 2
                index_list_rotor_2 (list with M*(M-1)/2 entries): rotor 2 indices that "interact" with the ones of rotor 1
            ----
        '''
        index_list_rotor_1 = []
        index_list_rotor_2 = []
        for i in range(self.My):
            for j in range(self.Mx):
                index_list_rotor_1, index_list_rotor_2 = self.get_indices_for_rotor_ij(i, j, index_list_rotor_1, index_list_rotor_2)

        return index_list_rotor_1, index_list_rotor_2

    def creat_new_ref_state(self, iter_number, ref_state, q):
        '''
            Computes: new reference state to ensure orthogonality among excited states and reference ground state

            ----
            Inputs:
                iter_number (int): number of iterations of self-consistent effective hamiltonian diagonalization
                ref_state (3-dimensional: (My,Mx,n)): reference ground state
                q (2-dimensional: (qx, qy)): momentum numbers of ref. ground state
            ----

            ----
            Description:
                1. Compute effective Hamiltonian and diagonalize it
                2. Create new reference ground state with the ground states of the diagonalizatoin
                3. Compute energy and overlap of the new ref ground state
                4. Repeat
            ----

            ----
            Outputs:
                new_ref (3-dimensional: (My,Mx,n)): new reference ground state
                conv_energ_gs (1-dimensional: (iter_number)): energies during the convergence
                overlap_arr (1-dimensional: (iter_number)): overlaps with previous state during the convergence
            ----
        '''

        new_ref = ref_state.copy()
        
        coupl_object = energy.coupling_of_states(params=self.param_dict)
        diag_object = diagonalization(params=self.param_dict)

        overlap_arr = np.zeros(iter_number, dtype=complex)
        conv_energ_gs = np.zeros(iter_number, dtype=complex)

        for t in range(iter_number):
            new_ref_next = new_ref.copy()
            energy_exc_states, psi_exc_states = diag_object.diag_h_eff(new_ref_next)

            for i in range(self.My):
                for j in range(self.Mx):
                    new_ref[i,j] = self.set_phase(psi_exc_states[i,j,0]) 
        
            conv_energ_gs[t] = coupl_object.calc_hamiltonian_matrix_element(new_ref, q, new_ref, q)[0]
            overlap_arr[t] = coupl_object.calc_overlap(new_ref, new_ref_next)
            print('Iter =', t, ', Overlap =', overlap_arr[t])
    
        return new_ref, conv_energ_gs, overlap_arr

    def append_single_excitation(self, ref_gs, psi_arr, psi_exc_states):
        '''
            Computes: single-excitation list

            ----
            Inputs:
                ref_gs (3-dimensional: (My,Mx,n)): reference ground state
                psi_arr (list): list of wavefunctions
                psi_excited_states (4-dimensional: (My,Mx,exc_states,n)): contains the excitation wavefunctions for every rotor 
            ----

            ----
            Description:
                Number of single-rotor excitations:
                    self.exc_states*self.My*self.Mx

                For every excitation number (max: self.exc_states), create a wavefunction wavefunction with 
                the i,j-th rotor excited in the respectively chosen excited state
            ----

            ----
            Outputs:
                psi_arr: updated with the single-rotor excitation wavefunctions
            ----
        '''
        for m in range(1,self.exc_states):
            for i in range(self.My):
                for j in range(self.Mx):
                    psi = ref_gs.copy() # to avoid pointer issues
                    psi[i,j] = self.set_phase(psi_exc_states[i,j,m]) 

                    psi_arr.append(psi)

        return psi_arr

    def append_double_excitations(self, ref_gs, psi_arr, psi_exc_states):
        '''
            Computes: double-excitation list

            ----
            Inputs:
                ref_gs (3-dimensional: (My,Mx,n)): reference ground state
                psi_arr (list): list of wavefunctions
                psi_excited_states (4-dimensional: (My,Mx,exc_states,n)): contains the excitation wavefunctions for every rotor 
            ----

            ----
            Description:
                Number of double-rotor excitations: from combinatorics:
                    self.exc_states**2*(self.My*self.Mx*(self.My*self.Mx-1)/2)

                For every excitation number (max: self.exc_states), create a wavefunction wavefunction with 
                the i,j-th rotor excited in the respectively chosen excited state
            ----

            ----
            Outputs:
                psi_arr: updated with the double-rotor excitation wavefunctions
            ----
        '''
        index_list_rotor_1, index_list_rotor_2 = self.get_double_rotor_excitation_list()
        inter_combinations = len(index_list_rotor_1)
        for m1 in range(1,self.exc_states):
            for m2 in range(1,self.exc_states):
                for i in range(inter_combinations):
                    index_rotor_1 = index_list_rotor_1[i]
                    index_rotor_2 = index_list_rotor_2[i]

                    psi = ref_gs.copy()

                    psi[index_rotor_1[0],index_rotor_1[1]] = self.set_phase(psi_exc_states[index_rotor_1[0],index_rotor_1[1],m1]) 
                    psi[index_rotor_2[0],index_rotor_2[1]] = self.set_phase(psi_exc_states[index_rotor_2[0],index_rotor_2[1],m2]) 

                    psi_arr.append(psi)

        return psi_arr