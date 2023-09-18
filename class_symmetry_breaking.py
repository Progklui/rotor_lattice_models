import numpy as np
import scipy 

import os, sys, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

import class_equations_of_motion as eom 
import class_handle_wavefunctions as h_wavef
import class_energy as energy

class symm_breaking:
    ''' Class for breaking the symmetry

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
        self.ext_field = -20*np.ones(self.Mx-2)

    def bias_potential(self, energy_object, psi1, psi2):
        psi1_conj = np.conjugate(psi1) 

        # np.cos(self.x+0.5*np.pi)
        # np.exp(-10*(self.x-0.5*np.pi)**2)

        E_bias = 0+0j
        for i in range(self.My):
            for j in range(1,self.Mx-1):
                #print(i,j)
                #print(self.ext_field[j-1])
                E_bias += self.ext_field[j-1]*np.sum(np.abs(np.cos(0.5*self.x-0.25*np.pi))*psi1_conj[i,j]*psi2[i,j])*energy_object.prod_excl_i_jth_rotor(psi1, psi2, i, j)

        return E_bias 

    def bulk_symmetry_breaking_mat_elements(self, psi1, psi2):
        psi1c = psi1.copy()
        psi2c = psi2.copy()

        coupl_object = energy.coupling_of_states(params=self.param_dict)
        energy_object = energy.energy(params=self.param_dict)

        q1 = np.array([0,0])
        q2 = np.array([0,0])

        E, E_T, E_B, E_V = coupl_object.calc_hamiltonian_matrix_element(psi1c, q1, psi2c, q2)
        E_bias = self.bias_potential(energy_object, psi1, psi2)
        
        E_new = E + E_bias

        return E_new, E_bias
    
    def calc_mat_bulk_symm_breaking(self, psi_arr, n_states):
        coupl_object = energy.coupling_of_states(params=self.param_dict)

        h_eff = np.zeros((n_states,n_states), dtype=complex)
        s_ove = np.zeros((n_states,n_states), dtype=complex)

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

                E12, E_bias12 = self.bulk_symmetry_breaking_mat_elements(psi1, psi2)
                overlap_12 = coupl_object.calc_overlap(psi1,psi2)
                
                h_eff[i,j] = E12
                s_ove[i,j] = overlap_12

                E_list[j] = E12
                S_list[j] = overlap_12
            #in_object.store_matrices_during_computation(self.V_0, E_list, S_list, path)

        return h_eff, s_ove