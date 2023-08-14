import numpy as np
import scipy 

import os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

import class_handle_wavefunctions as h_wavef
import class_handle_input as h_in

class energy:
    ''' Class for computing the energy of a given wavefunction

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
            self.calc_energy(wavefunc. as three-dimensional numpy array): calculate energy for a given psi and parameters given in psi
            self.deriv_dE_dt(wavefunc. as three-dimensional numpy array): computes partial derivative of E with respect to tx and ty
            self.analytic_small_polaron_energy(): calculate analytic small polaron energies
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
        self.n   = int(params['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid

    def calc_energy(self, psi):
        '''
            Computes: energy of psi

            ----
            Inputs:
                psi (3-dimensional: (My, Mx, n), dtype: complex): rotor wavefunction
            ----

            ----
            Variables: 
                psi_collection_conj (3-dimensional: (My, Mx, n), dtype: complex): complex conjugate

                TD_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for down jumping 
                TU_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for up jumping
                TR_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for right jumping
                TL_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for left jumping

                TD (scalar, dtype=complex): product of TD_arr
                TU (scalar, dtype=complex): product of TU_arr
                TR (scalar, dtype=complex): product of TR_arr
                TL (scalar, dtype=complex): product of TL_arr

                E_T (scalar, dtype=complex): kinetic/tunneling energy of electron
                E_B (scalar, dtype=complex): kinetic energy of rotors
                E_V (scalar, dtype=complex): interaction energy of electrons and rotors
                E (scalar, dtype=complex): total energy
                E_out (4-dimensional: (E, E_T, E_B, E_V)): energy array of psi
            ----

            ----
            Outputs:
                E_out (4-dimensional: (E, E_T, E_B, E_V)): energy array of psi
            ----
        '''

        psi = psi.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape

        '''
        tunneling energy
        '''
        dE_dtx, dE_dty = self.deriv_dE_dt(psi)
        E_T = self.ty*dE_dty + self.tx*dE_dtx

        '''
        rotor kinetic energy
        '''
        E_B = self.rotor_kinetic_energy(psi,psi)

        '''
        interaction energy
        '''
        E_V = self.V_0*np.sum(np.cos(self.x-0.25*np.pi)*np.abs(psi[self.My-1,0])**2)
        E_V += self.V_0*np.sum(np.cos(self.x-0.75*np.pi)*np.abs(psi[self.My-1,self.Mx-1])**2)
        E_V += self.V_0*np.sum(np.cos(self.x+0.25*np.pi)*np.abs(psi[0,0])**2)
        E_V += self.V_0*np.sum(np.cos(self.x+0.75*np.pi)*np.abs(psi[0,self.Mx-1])**2)
        
        '''
        total energy
        '''
        E = E_T + E_V + E_B 

        E_out = np.array([E, E_T, E_B, E_V], dtype=complex)
        return E_out 

    def deriv_dE_dt(self, psi):
        '''
            Computes: derivative of the energy with respect to tx and ty

            ----
            Inputs:
                psi (3-dimensional: (My, Mx, n), dtype: complex): rotor wavefunction
            ----

            ----
            Variables: 
                psi_collection_conj (3-dimensional: (My, Mx, n), dtype: complex): complex conjugate

                TD_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for down jumping 
                TU_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for up jumping
                TR_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for right jumping
                TL_arr (2-dimensional: (My, Mx), dtype=complex): transfer integral for left jumping

                TD (scalar, dtype=complex): product of TD_arr
                TU (scalar, dtype=complex): product of TU_arr
                TR (scalar, dtype=complex): product of TR_arr
                TL (scalar, dtype=complex): product of TL_arr

                dE_dtx (scalar, dtype=complex): partial derivative of E with respect to tx
                dE_dty (scalar, dtype=complex): partial derivative of E with respect to ty
            ----

            ----
            Outputs:
                dE_dtx, dE_dty
            ----
        '''
        psi = psi.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape
        
        '''
        compute transfer integrals
        '''
        TD, TU, TR, TL = self.transfer_integrals(psi,psi)

        '''
        partial derivatives
        '''
        dE_dtx = -(np.exp(-1j*2*np.pi*self.qx/self.Mx)*TR + np.exp(+1j*2*np.pi*self.qx/self.Mx)*TL)
        dE_dty = -(np.exp(-1j*2*np.pi*self.qy/self.My)*TD + np.exp(+1j*2*np.pi*self.qy/self.My)*TU)

        return dE_dtx, dE_dty

    def rotor_kinetic_energy(self, psi1, psi2):
        '''
            Computes: the projection of rotor kinetic energy between psi1 and psi2

            ----
            Inputs: 
                psi1 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 1
                psi2 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 2
            ----

            ----
            Outputs:
                E_B (scalar): rotor kinetic energy
            ----
        '''
        psi1c = psi1.copy()
        psi2c = psi2.copy()

        psi1_conj = np.conjugate(psi1c)

        #prod_arr = np.zeros((self.My,self.Mx), dtype=complex)
        #for i in range(self.My):
        #    for j in range(self.Mx):
        #        prod_arr[i,j] = self.prod_excl_i_jth_rotor(psi1, psi2, i, j)

        prod_arr = np.prod(np.einsum('ijk,ijk->ij', psi1_conj, psi2c))/np.einsum('ijk,ijk->ij', psi1_conj, psi2c)
        
        k2  = -np.append(np.arange(0,self.n/2+1),np.arange(-self.n/2+1,0))**2 # second derivative matrix

        single_rotor_deriv_sp = np.einsum('ijk,ijk->ij', psi1_conj, np.fft.ifft(k2*np.fft.fft(psi2c)))
        sum_elements = np.einsum('ij,ij->ij', single_rotor_deriv_sp, prod_arr)

        E_B = -self.B*np.sum(sum_elements) + 0j

        return E_B

    def prod_excl_i_jth_rotor(self, psi1, psi2, i, j):
        psi1c = psi1.copy()
        psi2c = psi2.copy()

        psi1c[i,j,:] = self.n**(-0.5)*np.ones((self.n,), dtype=complex)
        psi2c[i,j,:] = self.n**(-0.5)*np.ones((self.n,), dtype=complex)

        psi1c_conj = np.conjugate(psi1c)

        prod = np.prod(np.einsum('ijk,ijk->ij', psi1c_conj, psi2c)) #/np.sum(psi1_conj[i,j]*psi2[i,j])
        return prod
    
    def transfer_matrices(self, psi1, psi2):
        '''
            Computes: transfer matrices for two w.f. psi1 and psi2

            ----
            Inputs:
                psi1 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 1
                psi2 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 2
            ----

            ----
            Outputs:
                TD_arr (2-dimensional: (My,Mx)): products in matrix form of all single rotor transfer integrals for jumping down
                TU_arr (2-dimensional: (My,Mx)): products in matrix form of all single rotor transfer integrals for jumping up 
                TR_arr (2-dimensional: (My,Mx)): products in matrix form of all single rotor transfer integrals for jumping right
                TL_arr (2-dimensional: (My,Mx)): products in matrix form of all single rotor transfer integrals for jumping left
            ----
        '''
        psi1c = psi1.copy()
        psi2c = psi2.copy()

        psi1_conj = np.conjugate(psi1c)
        wfn2_manip = h_wavef.permute_rotors(psi2c)

        TD_arr = np.einsum('ijk,ijk->ij', psi1_conj, wfn2_manip.get_next_y_rotor(), dtype=complex)
        TU_arr = np.einsum('ijk,ijk->ij', psi1_conj, wfn2_manip.get_prev_y_rotor(), dtype=complex)
        TR_arr = np.einsum('ijk,ijk->ij', psi1_conj, wfn2_manip.get_next_x_rotor(), dtype=complex)
        TL_arr = np.einsum('ijk,ijk->ij', psi1_conj, wfn2_manip.get_prev_x_rotor(), dtype=complex)

        return TD_arr, TU_arr, TR_arr, TL_arr
    
    def transfer_integrals(self, psi1, psi2):
        '''
            Computes: product of transfer integrals for two w.f. psi1 and psi2

            ----
            Inputs:
                psi1 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 1
                psi2 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 2
            ----

            ----
            Outputs:
                TD (scalar): product of all single rotor transfer integrals for jumping down
                TU (scalar): product of all single rotor transfer integrals for jumping up 
                TR (scalar): product of all single rotor transfer integrals for jumping right
                TL (scalar): product of all single rotor transfer integrals for jumping left
            ----
        '''
        psi1_conj = np.conjugate(psi1)
        wfn2_manip = h_wavef.permute_rotors(psi2)

        TD_arr, TU_arr, TR_arr, TL_arr = self.transfer_matrices(psi1, psi2)
        
        TD = np.prod(TD_arr)
        TU = np.prod(TU_arr)
        TR = np.prod(TR_arr)
        TL = np.prod(TL_arr)

        return TD, TU, TR, TL
    
    def analytic_small_polaron_energy(self):
        '''
            NOTE: the evaluation of the wavefunctions doesn't work for very large values of mathieu_param, i.e. for small B's there are problems!
        '''
        mathieu_param = 2*self.V_0/self.B

        E = scipy.special.mathieu_a(0, mathieu_param)
        E += scipy.special.mathieu_a(0, mathieu_param)
        E += scipy.special.mathieu_a(0, mathieu_param)
        E += scipy.special.mathieu_a(0, mathieu_param)

        E *= self.B/4.

        return E

class coupling_of_states:
    ''' Class for computing the effective hamiltonian, diagonalization and transition probability calculation

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
            self.calc_energy(wavefunc. as three-dimensional numpy array): calculate energy for a given psi and parameters given in psi
            self.analytic_small_polaron_energy(): calculate analytic small polaron energies
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
        self.n   = int(params['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
        self.dt  = float(params['dt'])

    

    def calc_hamiltonian_matrix_element(self, psi1, q1, psi2, q2): 
        '''
            Computes: matrix element <psi1|H|psi2> 

            ----
            Inputs: 
                psi1 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 1
                psi2 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 2
                q1 (1-dimensional: (qx,qy)): momenta of psi1
                q2 (1-dimensional: (qx,qy)): momenta of psi2
            ----

            ----
            Outputs:
                E = <psi1|H|psi2> (scalar): overlap with respect to full hamiltonian
                E_T (scalar): overlap with respect to electron kinetic term
                E_B (scalar): overlap with respect to rotor kinetic term
                E_V (scalar): overlap with respect to interaction term
            ----
        '''
        psi1 = psi1.reshape((self.My, self.Mx, self.n)).copy()
        psi2 = psi2.reshape((self.My, self.Mx, self.n)).copy()
      
        qx1 = q1[0]
        qy1 = q1[1]

        psi1_conj = np.conjugate(psi1)

        energy_object = energy(params=self.param_dict)

        '''
        compute transfer integrals and plane wave factors
        '''
        TD, TU, TR, TL = energy_object.transfer_integrals(psi1, psi2)

        t_fac_TD = np.exp(-1j*(2*np.pi/self.My)*qy1)
        t_fac_TU = np.exp(+1j*(2*np.pi/self.My)*qy1)
        t_fac_TR = np.exp(-1j*(2*np.pi/self.Mx)*qx1)
        t_fac_TL = np.exp(+1j*(2*np.pi/self.Mx)*qx1)

        '''
        electron kinetic energy
        '''
        E_T = -self.ty*(t_fac_TD*TD + t_fac_TU*TU)\
            -self.tx*(t_fac_TR*TR + t_fac_TL*TL)
            
        '''
        rotor kinetic energy
        '''
        E_B = energy_object.rotor_kinetic_energy(psi1,psi2)

        '''
        interaction energy
        '''
        E_V = self.V_0*np.sum(np.cos(self.x-0.25*np.pi)*psi1_conj[self.My-1,0]*psi2[self.My-1,0])*energy_object.prod_excl_i_jth_rotor(psi1, psi2, self.My-1, 0)
        E_V += self.V_0*np.sum(np.cos(self.x-0.75*np.pi)*psi1_conj[self.My-1,self.Mx-1]*psi2[self.My-1,self.Mx-1])*energy_object.prod_excl_i_jth_rotor(psi1, psi2, self.My-1, self.Mx-1)
        E_V += self.V_0*np.sum(np.cos(self.x+0.25*np.pi)*psi1_conj[0,0]*psi2[0,0])*energy_object.prod_excl_i_jth_rotor(psi1, psi2, 0, 0)
        E_V += self.V_0*np.sum(np.cos(self.x+0.75*np.pi)*psi1_conj[0,self.Mx-1]*psi2[0,self.Mx-1])*energy_object.prod_excl_i_jth_rotor(psi1, psi2, 0, self.Mx-1)
        
        E = E_T + E_V + E_B # sum the individual energy contributions for total energy

        return E, E_T, E_B, E_V
    
    def calc_overlap(self, psi1, psi2):
        '''
            Computes: overlap of psi1, psi2, i.e. <psi1|psi2>

            ----
            Inputs:
                psi1 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 1
                psi2 (3-dimensional: (My,Mx,n), dtype=complex): rotor wavefunction 2
            ----

            ----
            Outputs:
                overlap (scalar): overlap matrix element 
            ----
        '''
        psi1c = psi1.reshape((self.My, self.Mx, self.n)).copy() 
        psi2c = psi2.reshape((self.My, self.Mx, self.n)).copy() 

        #overlap = 1 + 0j
        #for k in range(self.My): 
        #    for p in range(self.Mx):
        #        overlap *= np.sum(np.conjugate(psi1[k,p])*psi2[k,p])

        overlap = np.prod(np.einsum('ijk,ijk->ij', np.conjugate(psi1c), psi2c))
        return overlap
    
    def calc_hamiltonian(self, n_states, psi_arr, q_arr):
        '''
            Computes: hamilton and overlap matrix for the generalized eigenvalue problem

            ----
            Inputs:
                n_states (int scalar): number of states to consider
                psi_arr (list [] with n_states entries): list of psi's
                q_arr (2-dimensional (n_states, 2)): plane wave momenta
            ----

            ----
            Outputs:
                h_eff (2-dimensional: (n_states,n_states)): hamilton matrix
                s_ove (2-dimensional: (n_states,n_states)): overlap matrix
            ----
        '''
        h_eff = np.zeros((n_states,n_states), dtype=complex)
        s_ove = np.zeros((n_states,n_states), dtype=complex)

        in_object = h_in.coupl_states(params_calc=self.param_dict, params_wfs=self.param_dict)

        '''
        Loop over all psi_i and psi_j combinations

        ----
        Comment: could be simplified to just run over upper diagonal
        ----
        '''
        for i in range(n_states):
            E_list = np.zeros(n_states, dtype=complex)
            S_list = np.zeros(n_states, dtype=complex)
            for j in range(n_states):
                psi1 = psi_arr[i]
                psi2 = psi_arr[j]
                
                q1 = q_arr[i]
                q2 = q_arr[j]

                E12, E_T12, E_B12, E_V12 = self.calc_hamiltonian_matrix_element(psi1, q1, psi2, q2)
                overlap_12 = self.calc_overlap(psi1,psi2)
                
                h_eff[i,j] = E12
                s_ove[i,j] = overlap_12

                E_list[j] = E12
                S_list[j] = overlap_12

            in_object.n_states = n_states
            #in_object.store_matrices_during_computation(self.V_0, E_list, S_list, path)

        in_object.store_matrices_at_end_of_computation(self.V_0, h_eff, s_ove, path)
        return h_eff, s_ove

    def diag_hamiltonian(self, hamiltonian, overlap_matrix):
        '''
            Computes: diagonalization of the effective hamiltonian, i.e. solves generalized e-val problem
                    
            ----
            Comments:
                - we solve a generalized eigenvalue problem since the basis is not orthogonal
            ----

            ----
            Inputs:
                hamiltonian (2-dimensional: (n_states,n_states)): hamilton matrix
                overlap_matrix (2-dimensional: (n_states,n_states)): overlap matrix
            ----

            ----
            Outputs:
                eigen_values (1-dimensional: (n_states)): order (ascending) list of eigenvalues
                y_theory (2-dimensional: (n_states,n_states)): array of eigenvectors, in 2nd column (access through [:,number]), ordered
                e_vec1 (2-dimensional: (n_states,n_states)): array of overlap_matrix eigenvectors, in 2nd column (access through [:,number]), ordered
            ----
        '''

        '''
        Diagonalize the overlap matrix
        '''
        e_vals1, e_vec1 = np.linalg.eigh(overlap_matrix)
        order = np.argsort(e_vals1)
        e_vec1 = e_vec1[:,order]
        e_vals1 = e_vals1[order]
        #hamiltonian = np.linalg.inv(np.diag(e_vals1))@np.linalg.inv(e_vec1)@hamiltonian@e_vec1

        '''
        Solve the general eigenvalue problem: H|psi>=E*S*|psi>
        '''
        eigen_values, eigen_vector = scipy.linalg.eig(a=hamiltonian, b=overlap_matrix)
        order = np.argsort(eigen_values)
        eigen_vector = eigen_vector[:,order]
        eigen_values = eigen_values[order]

        '''
        Normalize the eigenvectors
        '''
        y_theory = np.zeros((len(eigen_values),len(eigen_values)), dtype=complex)
        for i in range(len(eigen_values)):
            y_theory[:,i] = eigen_vector[:,i].copy()/(np.sqrt(np.sum(np.conjugate(eigen_vector[:,i])*eigen_vector[:,i])))
            e_vec1[:,i] = e_vec1[:,i].copy()/(np.sqrt(np.sum(np.conjugate(e_vec1[:,i])*e_vec1[:,i])))
            
        return eigen_values.real, y_theory, e_vec1
    
    def transition_probabilities(self, n_states, e_kets, s_overlap):
        '''
            Computes: transition amplitudes 
                    
            ----
            Comments:
                - invovles working in a non-orthogonal basis -> look up notes for reference
            ----

            ----
            Inputs:
                n_states (scalar): number of states
                e_kets (2-dimensional: (n_states,n_states)): collection of eigenkets, access: j-th eigenvector = e_kets[:,j]
                s_overlap (2-dimensional: (n_states,n_states)): overlap matrix
            ----
            
            ----
            Important Variables:
                amp_j (1-dimensional: (n_states)): amp_j[i] gives the overlap of the i-th psi^old with psi_j^new, where j stands for the 
                                                ordered index of the new eigenstates
            ----

            ----
            Outputs:
                trans_probs (list): e.g. 1st entry: |<psi_i^old|\spi_0^{new,MRCI}>|^2 for all i
            ----
        '''

        trans_probs = []
        for j in range(n_states):
            '''
            Normalize the eigenvectors of the diagonalization
            '''
            ket_norm = np.sqrt(np.conjugate(e_kets[:,j].T)@s_overlap@e_kets[:,j])
            e_kets[:,j] = e_kets[:,j]/ket_norm

            '''
            Amplitudes of j-th state with respect to the old basis states
            '''
            amp_j = s_overlap@e_kets[:,j] 
            trans_prob_j = np.abs(amp_j)**2

            trans_probs.append(trans_prob_j)

        return trans_probs