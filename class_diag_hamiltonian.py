import numpy as np
import scipy 

import os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

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

    def second_derivative_mat(self):
        k1 = 1j*np.fft.ifftshift(np.arange(-self.n/2,self.n/2))
        
        k2 = -np.append(np.arange(0,self.n/2+1),np.arange(-self.n/2+1,0))**2 # make second derivative matrix
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
        return np.outer(psi[i1,j1], psi[i2,j2])
    
    def diag_h_eff(self, psi):
        '''
        objects to store the excitation wavefunctions and e-vals
        '''
        psi_exc_states = np.zeros((self.My,self.Mx,self.exc_states,self.n), dtype=complex) 
        energy_exc_states = np.zeros((self.My,self.Mx,self.exc_states), dtype=complex)

        energy_object = energy.energy(params=self.param_dict)
        sec_derivative = self.second_derivative_mat() 

        TD, TU, TR, TL = energy_object.transfer_integrals(psi, psi)
        TD_arr, TU_arr, TR_arr, TL_arr = energy_object.transfer_matrices(psi, psi)

        '''
        for every rotor compute H_psi, diagonalize it to get wave function and return result
        '''
        for i in range(self.My):
            for j in range(self.Mx):
                #print('Rotor ', i, j)

                ''' 
                rotor kinetic term
                '''
                H_psi = -self.B*sec_derivative 
            
                '''
                transfer terms
                '''
                TDr = TD / (TD_arr[i, j]*TD_arr[i-1, j])
                TUr = TU / (TU_arr[i, j]*TD_arr[(i+1)%self.My, j])
                TRr = TR / (TR_arr[i, j]*TD_arr[i, j-1])
                TLr = TL / (TL_arr[i, j]*TD_arr[i, (j+1)%self.Mx])

                H_psi -= self.ty*(np.exp(-1j*(2*np.pi*self.qy/self.My))*TDr*self.projector(psi, (i+1)%self.My, j, i-1, j)\
                            + np.exp(+1j*(2*np.pi*self.qy/self.My))*TUr*self.projector(psi, i-1, j, (i+1)%self.My, j))
                H_psi -= self.tx*(np.exp(-1j*(2*np.pi*self.qx/self.Mx))*TRr*self.projector(psi, i, (j+1)%self.Mx, i, j-1) \
                            + np.exp(+1j*(2*np.pi*self.qx/self.Mx))*TLr*self.projector(psi, i, j-1, i, (j+1)%self.Mx))

                '''
                interaction terms for right rotors
                '''
                if i == (self.My-1) and j == 0: 
                    H_psi += np.diag(self.V_0*np.cos(self.x-0.25*np.pi))
                elif i == (self.My-1) and j == (self.Mx-1): 
                    H_psi += np.diag(self.V_0*np.cos(self.x-0.75*np.pi))
                elif i == 0 and j == 0: 
                    H_psi += np.diag(self.V_0*np.cos(self.x+0.25*np.pi))
                elif i == 0 and j == (self.Mx-1): 
                    H_psi += np.diag(self.V_0*np.cos(self.x+0.75*np.pi))

                '''
                diagonalization of h_eff
                '''
                eigen_values, eigen_vector = np.linalg.eigh(H_psi) 
                order = np.argsort(eigen_values)
                eigen_vector = eigen_vector[:,order]
                eigen_values = eigen_values[order]

                #print('e-vals =', eigen_values[0:self.exc_states])

                for n in range(self.exc_states):
                    psi_exc_n = self.get_state_i(eigen_vector, n)

                    psi_exc_states[i,j,n] = psi_exc_n
                    energy_exc_states[i,j,n] = eigen_values[n]

        return energy_exc_states, psi_exc_states