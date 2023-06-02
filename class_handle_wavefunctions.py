import numpy as np
from scipy.integrate import solve_ivp
import scipy 

import os, sys

path = os.path.dirname(__file__) 
sys.path.append(path)

class wavefunctions:
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

    def create_init_wavefunction(self, phase):
        # psi_init is expected in the format Mx*My*n
        psi_init = self.n**(-0.5)*np.ones((int(self.Mx*self.My)*self.n),dtype=complex)
        psi_init = psi_init.reshape((self.My,self.Mx,self.n)).copy()

        if phase == 'uniform':
            psi_init = self.n**(-0.5)*np.ones((int(self.Mx*self.My)*self.n),dtype=complex)
        
        # initialization for the ferroelectric domain, wall oriented vertically
        elif phase == 'ferro_domain_vertical_wall': 
            for i in range(self.My):
                # other option is to try with np.cos(0.5*x) and np.sin(...)

                # left column
                psi_init[i,self.Mx-1] = np.cos(self.x) + 0j
                psi_init[i,self.Mx-1][int(self.n/4):int(self.n/2)] = 0.01 + 0j
                psi_init[i,self.Mx-1][int(self.n/2):int(3*self.n/4)] = 0.01 + 0j

                # right column
                psi_init[i,0] = np.cos(self.x) + 0j
                psi_init[i,0][0:int(self.n/4)] = 0.01 + 0j
                psi_init[i,0][int(3*self.n/4):self.n] = 0.01 + 0j
                
        # initialization for the ferroelectric domain, wall oriented horizontally   
        elif phase == 'ferro_domain_horizontal_wall': 
            for j in range(self.Mx):
                # top row
                psi_init[self.My-1,j] = np.sin(self.x) 
                psi_init[self.My-1,j][0:int(self.n/2)] = 0.01 

                # bottom row
                psi_init[0,j] = np.sin(self.x) 
                psi_init[0,j][int(self.n/2):self.n] = 0.01 
        
        # random initialization
        elif phase == 'random': 
            # psi_init = self.n**(-0.5)*np.ones((self.My,self.Mx,self.n),dtype=complex) + 0.02*np.random.rand(self.My,self.Mx,self.n) # np.random.rand(self.My,self.Mx,self.n)
            # psi_init = n**(-0.5)*np.ones((My,Mx,n),dtype=complex) + 0.02*np.random.rand(My,Mx,n) # that's the old version - not connected points
            for i in range(self.My):
                for j in range(self.Mx):
                    H = 10
                    rho = np.random.rand(1,H)*np.logspace(-0.5,-2.5,H)
                    phi = np.random.rand(1,H)*2*np.pi

                    # Accumulate r(t) over t=[0,2*pi]
                    t = (2*np.pi/self.n)*np.arange(self.n) # np.linspace(0,2*np.pi,n)
                    r = np.ones(len(t))
                    for h in range(H):
                        r = r + rho[0][h]*np.ones(len(t))*np.sin(h*t+phi[0][h]*np.ones(len(t)))

                    # Reconstruct x(t), y(t)
                    x = r*np.cos(t)
                    y = r*np.sin(t)

                    psi_init[i,j] = r + r*1j # not entirely sure about the imaginary part here

        # analytic small polaron wavefunctions
        elif phase == 'small_polaron': 
            # bottom left
            y, yp = scipy.special.mathieu_cem(0, 2*self.V_0/self.B, (self.x+3*np.pi/4)/2*180/np.pi)
            psi_init[0,self.Mx-1] = y/np.sqrt(np.sum(y*y))
        
            # bottom right
            y, yp = scipy.special.mathieu_cem(0, 2*self.V_0/self.B, (self.x+np.pi/4)/2*180/np.pi)
            psi_init[0,0] = y/np.sqrt(np.sum(y*y))

            # top left
            y, yp = scipy.special.mathieu_cem(0, 2*self.V_0/self.B, (self.x-3*np.pi/4)/2*180/np.pi)
            psi_init[self.My-1,self.Mx-1] = y/np.sqrt(np.sum(y*y))

            # top right
            y, yp = scipy.special.mathieu_cem(0, 2*self.V_0/self.B, (self.x-np.pi/4)/2*180/np.pi)
            psi_init[self.My-1,0] = y/np.sqrt(np.sum(y*y))

        else: # sanity check
            return
        
        # normalize wave functions
        normalization_factor = (1.0/np.sqrt(np.sum(np.abs(psi_init.reshape((int(self.Mx*self.My),self.n)))**2,axis=1))).reshape(int(self.Mx*self.My),1)
        psi = normalization_factor*psi_init.reshape((int(self.Mx*self.My),self.n))
        return psi.reshape((int(self.Mx*self.My)*self.n)) # reshaping the array