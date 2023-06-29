import numpy as np
import scipy 

import os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

import class_handle_wavefunctions as h_wavef

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

        # tunneling energy
        dE_dtx, dE_dty = self.deriv_dE_dt(psi)
        E_T = self.ty*dE_dty + self.tx*dE_dtx

        # kinetic energy of rotors
        '''
            outsource the computation of the second derivative 
        '''
        k2  = -np.append(np.arange(0,self.n/2+1),np.arange(-self.n/2+1,0))**2 # make second derivative matrix
        E_B = -self.B*np.sum(np.einsum('ijk,ijk->ij', np.conjugate(psi), np.fft.ifft(k2*np.fft.fft(psi))))  + 0j
        #for k in range(self.My):
        #    for p in range(self.Mx):
        #        E_B -= self.B*np.sum(np.conjugate(psi[k,p])*np.fft.ifft(k2*np.fft.fft(psi[k,p])))

        # interaction energy
        E_V = self.V_0*np.sum(np.cos(self.x-0.25*np.pi)*np.abs(psi[self.My-1,0])**2)
        E_V += self.V_0*np.sum(np.cos(self.x-0.75*np.pi)*np.abs(psi[self.My-1,self.Mx-1])**2)
        E_V += self.V_0*np.sum(np.cos(self.x+0.25*np.pi)*np.abs(psi[0,0])**2)
        E_V += self.V_0*np.sum(np.cos(self.x+0.75*np.pi)*np.abs(psi[0,self.Mx-1])**2)
        
        E = E_T + E_V + E_B # sum the individual energy contributions for total energy

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
        
        # object for manipulating wavefunctions
        wfn_manip = h_wavef.permute_rotors(psi)

        psi_collection_conj = np.conjugate(psi)

        # compute transfer integrals 
        TD_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_next_y_rotor(), dtype=complex)
        TU_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_prev_y_rotor(), dtype=complex)
        TR_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_next_x_rotor(), dtype=complex)
        TL_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_prev_x_rotor(), dtype=complex)
        
        TD = np.prod(TD_arr)
        TU = np.prod(TU_arr)
        TR = np.prod(TR_arr)
        TL = np.prod(TL_arr)

        # partial derivatives
        dE_dtx = -(np.exp(-1j*2*np.pi*self.qx/self.Mx)*TR + np.exp(+1j*2*np.pi*self.qx/self.Mx)*TL)
        dE_dty = -(np.exp(-1j*2*np.pi*self.qy/self.My)*TD + np.exp(+1j*2*np.pi*self.qy/self.My)*TU)

        return dE_dtx, dE_dty

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
    ''' Class for computing the effective hamiltonian

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

    # calculate matrix element <psi1|H|psi2>  
    def calc_hamiltonian_matrix_element(self, psi1, q1, psi2, q2): 
        psi1 = psi1.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape
        psi2 = psi2.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape
      
        qx1 = q1[0]
        qy1 = q1[1]

        qx2 = q2[0]
        qy2 = q2[1]

        # compute all the transfer integrals
        TD = 1 + 0j; TU = 1 + 0j; TR = 1 + 0j; TL = 1 + 0j
        for k in range(self.My): 
            for p in range(self.Mx):
                TD = TD*np.sum(np.conjugate(psi1[k,p])*psi2[(k+1)%self.My,p])
                TU = TU*np.sum(np.conjugate(psi1[k,p])*psi2[k-1,p])
                
                TR = TR*np.sum(np.conjugate(psi1[k,p])*psi2[k,(p+1)%self.Mx])
                TL = TL*np.sum(np.conjugate(psi1[k,p])*psi2[k,p-1])

        t_fac_TD = np.exp(-1j*(2*np.pi/self.My)*qy1)
        t_fac_TU = np.exp(+1j*(2*np.pi/self.My)*qy1)
        t_fac_TR = np.exp(-1j*(2*np.pi/self.Mx)*qx1)
        t_fac_TL = np.exp(+1j*(2*np.pi/self.Mx)*qx1)

        # tunneling energy
        E_T = -self.ty*(t_fac_TD*TD + t_fac_TU*TU)\
            -self.tx*(t_fac_TR*TR + t_fac_TL*TL)
            
        # kinetic energy of rotors
        k2  = -np.append(np.arange(0,self.n/2+1),np.arange(-self.n/2+1,0))**2 # make second derivative matrix
        E_B = 0+0j
        for k in range(self.My):
            for p in range(self.Mx):
                E_B -= self.B*np.sum(np.conjugate(psi1[k,p])*np.fft.ifft(k2*np.fft.fft(psi2[k,p])))

        # interaction energy
        E_V = self.V_0*np.sum(np.cos(self.x-0.25*np.pi)*np.conjugate(psi1[self.My-1,0])*psi2[self.My-1,0])
        E_V += self.V_0*np.sum(np.cos(self.x-0.75*np.pi)*np.conjugate(psi1[self.My-1,self.Mx-1])*psi2[self.My-1,self.Mx-1])
        E_V += self.V_0*np.sum(np.cos(self.x+0.25*np.pi)*np.conjugate(psi1[0,0])*psi2[0,0])
        E_V += self.V_0*np.sum(np.cos(self.x+0.75*np.pi)*np.conjugate(psi1[0,self.Mx-1])*psi2[0,self.Mx-1])
        
        E = E_T + E_V + E_B # sum the individual energy contributions for total energy

        return E, E_T, E_B, E_V
    
    # calc overlap <psi1|H|psi2>
    def calc_overlap(self, psi1, psi2):
        psi1 = psi1.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape
        psi2 = psi2.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape

        overlap = 1 + 0j
        for k in range(self.My): 
            for p in range(self.Mx):
                overlap *= np.sum(np.conjugate(psi1[k,p])*psi2[k,p])
        return overlap
    
    # compute the matrices for the generalized eigenvalue problem
    def calc_hamiltonian(self, n_states, psi_arr, q_arr):
        h_eff = np.zeros((n_states,n_states), dtype=complex)
        s_ove = np.zeros((n_states,n_states), dtype=complex)

        for i in range(n_states):
            for j in range(n_states):
                psi1 = psi_arr[i]
                psi2 = psi_arr[j]
                
                q1 = q_arr[i]
                q2 = q_arr[j]

                E12, E_T12, E_B12, E_V12 = self.calc_hamiltonian_matrix_element(psi1, q1, psi2, q2)
                overlap_12 = self.calc_overlap(psi1,psi2)
                
                h_eff[i,j] = E12
                s_ove[i,j] = overlap_12

        return h_eff, s_ove

    # diagonalize the effective hamiltonian
    def diag_hamiltonian(self, hamiltonian, overlap_matrix):
        e_vals1, e_vec1 = np.linalg.eigh(overlap_matrix)
        order = np.argsort(e_vals1)
        e_vec1 = e_vec1[:,order]
        e_vals1 = e_vals1[order]
        #hamiltonian = np.linalg.inv(np.diag(e_vals1))@np.linalg.inv(e_vec1)@hamiltonian@e_vec1

        eigen_values, eigen_vector = scipy.linalg.eig(a=hamiltonian, b=overlap_matrix) # diagonalize effective hamiltonian
        order = np.argsort(eigen_values)
        eigen_vector = eigen_vector[:,order]
        eigen_values = eigen_values[order]

        y_theory = np.zeros((len(eigen_values),len(eigen_values)), dtype=complex)
        for i in range(len(eigen_values)):
            y_theory[:,i] = eigen_vector[:,i].copy()/(np.sqrt(np.sum(np.conjugate(eigen_vector[:,i])*eigen_vector[:,i]))) # get ground state 
            e_vec1[:,i] = e_vec1[:,i].copy()/(np.sqrt(np.sum(np.conjugate(e_vec1[:,i])*e_vec1[:,i])))
            
        return eigen_values.real, y_theory, e_vec1
    
    def transition_probabilities(self, n_states, e_kets, s_overlap):
        # compute the transition amplitudes from every state to the next
        trans_probs = []
        for j in range(n_states):
            ket_norm = np.sqrt(np.conjugate(e_kets[:,j].T)@s_overlap@e_kets[:,j])
            e_kets[:,j] = e_kets[:,j]/ket_norm
        
            amp_j = s_overlap@e_kets[:,j] # amplitudes for the j-th state
            trans_prob_j = np.abs(amp_j)**2

            trans_probs.append(trans_prob_j)

        return trans_probs
