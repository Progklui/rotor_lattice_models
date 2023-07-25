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

                energ_object = energy.energy(params=self.param_dict)
                energ_object.qx = 0
                energ_object.qy = 0

                e_psi = energ_object.calc_energy(psi)[0]

                #H_psi -= np.diag(e_psi*np.ones(self.n, dtype=complex))

                mul_left = np.diag(np.ones(self.n, dtype=complex)) - np.outer(psi[i,j], psi[i,j])
                #H_psi = mul_left@H_psi

                lagrange_multiplier = np.conjugate(psi[i,j])*H_psi #@psi[i,j]  # np.einsum('ijk,ijk->ij', np.conjugate(psi), H_psi)
                #print(lagrange_multiplier)
                sign = np.sum((1/(self.n)**0.5)*psi)
                sign = sign/np.abs(sign)
                #return np.conjugate(sign)*psi
                #print(sign)
                lag_mul = np.diag(np.einsum('n,nn->n',np.conjugate(psi[i,j].T),H_psi))
                #print(lagrange_multiplier.shape)
                #print(H_psi.shape)
                #print(lag_mul.shape)
                #H_psi -= lagrange_multiplier #e_psi*np.ones(self.n, dtype=complex) #lagrange_multiplier #* np.einsum('n,nn->n', lag_mul, np.outer(psi[i,j],psi[i,j])) #np.diag(psi[i,j]) #e_psi*(1/self.n**0.5)*np.diag(np.ones(self.n, dtype=complex)) #e_psi*np.outer(psi[i,j], psi[i,j]) #np.diag(lagrange_multiplier*np.ones(self.n, dtype=complex)) # [:, :, np.newaxis] * psi_collection

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
    
class multi_ref_ci:
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

            psi_man = np.conjugate(sign)*psi_man

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