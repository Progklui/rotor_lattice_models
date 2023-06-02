import numpy as np
import scipy 

import os, sys

path = os.path.dirname(__file__) 
sys.path.append(path)

class energy:
    def __init__(self, Mx, My, B, V_0, tx, ty, qx, qy, n, x, dt, tol):
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.V_0 = V_0 # not an array expected here! - compute individually for the whole list
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.x   = x
        self.dt  = dt
        self.tol = tol

    def calc_energy(self, psi): # calculate the energy of every coupling
        psi = psi.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape

        # compute all the transfer integrals
        TD = 1 + 0j; TU = 1 + 0j; TR = 1 + 0j; TL = 1 + 0j
        for k in range(self.My): 
            for p in range(self.Mx):
                TD = TD*np.sum(np.conjugate(psi[k,p])*psi[(k+1)%self.My,p])
                TU = TU*np.sum(np.conjugate(psi[k,p])*psi[k-1,p])
                
                TR = TR*np.sum(np.conjugate(psi[k,p])*psi[k,(p+1)%self.Mx])
                TL = TL*np.sum(np.conjugate(psi[k,p])*psi[k,p-1])

        # tunneling energy
        E_T = -self.ty*(np.exp(-1j*2*np.pi*self.qy/self.My)*TD + np.exp(+1j*2*np.pi*self.qy/self.My)*TU) \
            -self.tx*(np.exp(-1j*2*np.pi*self.qx/self.Mx)*TR + np.exp(+1j*2*np.pi*self.qx/self.Mx)*TL)

        # kinetic energy of rotors
        k2  = -np.append(np.arange(0,self.n/2+1),np.arange(-self.n/2+1,0))**2 # make second derivative matrix
        E_B = 0+0j
        for k in range(self.My):
            for p in range(self.Mx):
                E_B -= self.B*np.sum(np.conjugate(psi[k,p])*np.fft.ifft(k2*np.fft.fft(psi[k,p])))

        # interaction energy
        E_V = self.V_0*np.sum(np.cos(self.x-0.25*np.pi)*np.abs(psi[self.My-1,0])**2)
        E_V += self.V_0*np.sum(np.cos(self.x-0.75*np.pi)*np.abs(psi[self.My-1,self.Mx-1])**2)
        E_V += self.V_0*np.sum(np.cos(self.x+0.25*np.pi)*np.abs(psi[0,0])**2)
        E_V += self.V_0*np.sum(np.cos(self.x+0.75*np.pi)*np.abs(psi[0,self.Mx-1])**2)
        
        E = E_T + E_V + E_B # sum the individual energy contributions for total energy

        return E, E_T, E_B, E_V # deliberately not converted to real - facilitates/allows error checking!

class coupling_of_states:
    def __init__(self, Mx, My, B, V_0, tx, ty, n, x, dt, tol):
        self.Mx   = Mx
        self.My   = My
        self.B    = B
        self.V_0  = V_0 # not an array expected here - compute individually for the whole list
        self.tx   = tx
        self.ty   = ty
        self.n    = n
        self.x    = x
        self.dt   = dt
        self.tol  = tol

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

    