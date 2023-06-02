import numpy as np
from scipy.optimize import fsolve

import os, sys, gc

import class_energy as energy
import class_equations_of_motion as eom

path = os.path.dirname(__file__) 
sys.path.append(path)

class eff_mass:
    def __init__(self, Mx, My, B, V_0, tx, ty, qx, qy, n, x, dt, tol):
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.V_0 = V_0
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.x   = x
        self.dt  = dt
        self.tol = tol
        self.mx_y = 'mx'

    def eff_mass(self, E_q1, E_q2, E_q3):
        if self.m_xy == 'mx':
            eff_m = ((E_q1-2*E_q2+E_q3)/(2*np.pi/self.Mx)**2)**(-1)
            m0    = ((self.tx*self.Mx**2/np.pi**2)*(1-np.cos(2*np.pi/self.Mx)))**(-1)
        elif self.m_xy == 'my':
            eff_m = ((E_q1-2*E_q2+E_q3)/(2*np.pi/self.My)**2)**(-1)
            m0    = ((self.ty*self.My**2/np.pi**2)*(1-np.cos(2*np.pi/self.My)))**(-1)
        return eff_m, m0

    def calc_eff_mass(self, psi, psi_qp, psi_qm, qx, qy): # calculate the energy of every coupling
        M = int(self.My*self.Mx)

        energy_object = energy.energy(Mx=self.Mx, My=self.My, B=self.B, V_0=0, tx=self.tx, ty=self.ty,
                                      qx=self.qx, qy=self.qy, n=self.n, x=self.x, dt=self.dt, tol=self.tol)

        if qx == 1 and qy == 0:
            self.m_xy = 'mx'
        elif qy == 1 and qx == 0:
            self.m_xy = 'my'
        else:
            raise Exception('Only qx == 1 and qy == 0 or qy == 1 and qx == 0 are supported')

        E_col = np.zeros((psi.shape[0],), dtype=complex)
        E_col_qp = np.zeros((psi_qp.shape[0],), dtype=complex) # for q = 1
        E_col_qm = np.zeros((psi_qm.shape[0],), dtype=complex) # for q = -1 - not needed, see comment below and latex file

        # q = 0
        energy_object.qx = 0 
        energy_object.qy = 0
        for i in range(psi.shape[0]):
            energy_object.V_0 = self.V_0[i]
            E_col[i], _, _, _ = energy_object.calc_energy(psi[i])#[0]

        # q = 1
        energy_object.qx = qx 
        energy_object.qy = qy

        for i in range(psi_qp.shape[0]):
            energy_object.V_0 = self.V_0[i]
            E_col_qp[i], _, _, _  = energy_object.calc_energy(psi_qp[i])#[0] 

        # q = -1
        # Important note: calculation wouldn't make sense - wrong wavefunction, BUT: result not dependend on sign of q!!

        m_eff, m_0 = self.eff_mass(E_col_qp.copy(), E_col.copy(), E_col_qp.copy())
        return m_eff.real, m_0, E_col.real, E_col_qp.real

class polaron_size:
    def __init__(self, Mx, My, B, V_0, tx, ty, qx, qy, n, x, dt, tol):
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.V_0 = V_0 # expects an array here
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.x   = x
        self.dt  = dt
        self.tol = tol

    def function_for_solver(self, alpha, distribution): # distribution should be normalized
        dx = self.x[1]-self.x[0]
        w = 2*np.pi*np.fft.fftfreq(self.x.size, dx)

        shifted_distribution = np.fft.ifft(np.exp(1.0j*np.angle(alpha-np.pi)*w)*np.fft.fft(distribution))
        return np.sum((self.x-np.pi) * shifted_distribution)

    def average_and_variance_over_shifting(self, distribution):
        dx = self.x[1]-self.x[0]
        w  = 2*np.pi*np.fft.fftfreq(self.x.size, dx)

        x0 = fsolve(lambda x0test: self.function_for_solver(x0test, distribution), 0)[0] # average
        shifted_distribution = np.fft.ifft(np.exp(1.0j*np.angle(x0-np.pi)*w)*np.fft.fft(distribution))

        var = np.sum((self.x-np.pi)**2 * shifted_distribution)

        #mean = np.sum(self.x*distribution)
        #var = np.sum((self.x-mean)**2 * distribution)
        return var/(np.pi**2/3) # 1-np.abs(1-var/(np.pi**2/3))

    def polaron_size(self, rotor_density_i_j, calculation_choice):
        if calculation_choice == '1':
            sigma_i_j_V = 1 - np.sqrt((np.sum(np.cos(self.x)*rotor_density_i_j))**2 + (np.sum(np.sin(self.x)*rotor_density_i_j))**2)

        elif calculation_choice == '2':
            sigma_i_j_V = self.average_and_variance_over_shifting(rotor_density_i_j)

        elif calculation_choice == '3':
            rotor_density_i_j = rotor_density_i_j
            sin_avg = np.sum(np.sin(self.x)*rotor_density_i_j).real
            cos_avg = np.sum(np.cos(self.x)*rotor_density_i_j).real

            mean_i_j = np.arctan2(sin_avg, cos_avg) # np.sign(sin_avg)*np.arccos(cos_avg/(np.sqrt(sin_avg**2 + cos_avg**2)))
            diff = np.arctan2(np.sin(self.x-mean_i_j),np.cos(self.x-mean_i_j))

            var = np.sum(rotor_density_i_j*diff**2)
            sigma_i_j_V = var/(np.pi**2/3)

        return sigma_i_j_V

    # takes a (pot_points, My, Mx, n) array and calculates the for a selected potential the polaron size for all rotors
    def calc_polaron_size(self, psi, V_index, calc_choice):
        sigma = np.zeros((self.My,self.Mx))
        for i in range(self.My):
            for j in range(self.Mx):
                rotor_density_i_j = ((np.conjugate(psi[:,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx])\
                    *psi[:,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx]))[V_index]

                sigma[i,j] = self.polaron_size(rotor_density_i_j, calc_choice).real

                del rotor_density_i_j
                gc.collect()
        return sigma

    # takes a (pot_points, My, Mx, n) array and calculates the for a selected potential the polaron size for all rotors
    def calc_polaron_size_real_time_prop(self, psi, calc_choice):
        sigma = np.zeros((self.My,self.Mx))
        for i in range(self.My):
            for j in range(self.Mx):
                rotor_density_i_j = ((np.conjugate(psi[:,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx])\
                    *psi[:,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx]))

                sigma[i,j] = self.polaron_size(rotor_density_i_j, calc_choice).real

                del rotor_density_i_j
                gc.collect()
        return sigma
    
    # takes a (My, Mx, n) array and calculates the for a selected potential the polaron size for all rotors
    def calc_polaron_size_section(self, psi, calc_choice):
        sigma = np.zeros((self.My,self.Mx))
        for i in range(self.My):
            for j in range(self.Mx):
                rotor_density_i_j = ((np.conjugate(psi[(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx])\
                    *psi[(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx]))

                sigma[i,j] = self.polaron_size(rotor_density_i_j, calc_choice).real

                del rotor_density_i_j
                gc.collect()
        return sigma
    
    def calc_polaron_size_along_ray(self, sigma, ray_indices, start_indices):
        sigma_new = []
        for i in range(np.max(np.array([sigma.shape[0], sigma.shape[1]]))):
            x = int(start_indices[0] + i*ray_indices[0])
            y = int(start_indices[1] + i*ray_indices[1])
            
            if x < sigma.shape[0] and y < sigma.shape[1]:
                sigma_new.append(sigma[x, y])

        return np.array(sigma_new)