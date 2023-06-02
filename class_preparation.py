import numpy as np
import scipy 

import matplotlib.pyplot as plt

import os, sys, csv

path = os.path.dirname(__file__) 
sys.path.append(path)

import class_handle_input as h_in
import class_energy as energy

class phase_transition:
    def __init__(self, psi_rotors_f1, psi_rotors_b1, psi_rotors_f2, psi_rotors_b2, 
                 potential_points1, potential_points2, pot_points_back1, pot_points_back2, Mx, My, B, tx, ty, qx, qy, n, x, dt, tol):
        self.psi_rotors_f1 = psi_rotors_f1
        self.psi_rotors_b1 = psi_rotors_b1
        self.psi_rotors_f2 = psi_rotors_f2
        self.psi_rotors_b2 = psi_rotors_b2
        self.potential_points1 = potential_points1
        self.potential_points2 = potential_points2
        self.pot_points_back1  = pot_points_back1
        self.pot_points_back2  = pot_points_back2
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.x   = x
        self.dt  = dt
        self.tol = tol

    def concatenate_wave_functions_2_phases(self, V_0_pool_f1, V_0_pool_b1, last_p1, first_p2):
        psi_phase1 = self.psi_rotors_f1.reshape(self.potential_points1, self.My, self.Mx, self.n).copy()[0:last_p1]
        psi_phase2 = self.psi_rotors_b1.reshape(self.potential_points1, self.My, self.Mx, self.n).copy()[first_p2:len(self.psi_rotors_b1)]

        V1 = V_0_pool_f1[0:last_p1]
        V2 = V_0_pool_b1[first_p2:len(self.psi_rotors_b1)]

        return V1.copy(), V2.copy(), psi_phase1.copy(), psi_phase2.copy()

    def get_wave_function_indices_2_phases(self, V_0_pool_f1, V_0_pool_b1):
        last_p1  = len(V_0_pool_f1)
        first_p2 = 0
    
        V1, V2, psi_phase1, psi_phase2 = self.concatenate_wave_functions_2_phases(V_0_pool_f1, V_0_pool_b1, last_p1, first_p2)

        energy_object = energy.energy(Mx=self.Mx, My=self.My, B=self.B, V_0=0, tx=self.tx, ty=self.ty,
                                      qx=self.qx, qy=self.qy, n=self.n, x=self.x, dt=self.dt, tol=self.tol)

        E_col_1 = np.zeros(len(V1))
        E_col_2 = np.zeros(len(V2))

        for i in range(len(V1)):
            energy_object.V_0 = V1[i]
            E_col_1[i] = energy_object.calc_energy(psi_phase1[i])[0].real
        for i in range(len(V2)):
            energy_object.V_0 = V2[i]
            E_col_2[i] = energy_object.calc_energy(psi_phase2[i])[0].real

        j = 0
        for i in range(int(np.where(V1 == V2[0])[0][0]), len(V1)-1):
            if E_col_2[j]+1 < E_col_1[i]:
                last_index_V1  = i
                first_index_V2 = j
                break
            j += 1

        V1 = V1[0:last_index_V1]
        V2 = V2[first_index_V2:len(V2)]

        psi_phase1 = psi_phase1[0:last_index_V1]
        psi_phase2 = psi_phase2[first_index_V2:len(psi_phase2)]
    
        return V1.copy(), V2.copy(), psi_phase1.copy(), psi_phase2.copy()

    def concatenate_wave_functions_3_phases(self, V_0_pool_f1, V_0_pool_f2, V_0_pool_b1, V_0_pool_b2, last_p1, first_p2, last_p2, first_p3):
        psi_phase1 = self.psi_rotors_f1.reshape(self.potential_points1, self.My,self.Mx,self.n).copy()[0:last_p1]
        
        psi_phase2 = self.psi_rotors_b1.reshape(self.pot_points_back1,self.My,self.Mx,self.n).copy()[first_p2:len(self.psi_rotors_b1)]
        psi_phase2 = np.concatenate((psi_phase2,self.psi_rotors_f2.reshape(self.potential_points2, self.My,self.Mx,self.n)[0:last_p2]),axis=0) #.reshape(potential_points2, My, Mx, n).copy()

        psi_phase3 = self.psi_rotors_b2.reshape(self.pot_points_back2, self.My,self.Mx,self.n).copy()[first_p3:len(self.psi_rotors_f2)]

        V1 = V_0_pool_f1[0:last_p1]
        V2 = np.concatenate((V_0_pool_b1[first_p2:len(V_0_pool_b1)], V_0_pool_f2[0:last_p2]))
        V3 = V_0_pool_b2[first_p3:len(self.psi_rotors_f2)]

        return V1.copy(), V2.copy(), V3.copy(), psi_phase1.copy(), psi_phase2.copy(), psi_phase3.copy()

    def get_wave_function_indices_3_phases(self, V_0_pool_f1, V_0_pool_f2, V_0_pool_b1, V_0_pool_b2):
        last_p1  = len(V_0_pool_f1)
        first_p2 = 0
        last_p2  = len(V_0_pool_f2)
        first_p3 = 0
    
        V1, V2, V3, psi_phase1, psi_phase2, psi_phase3 = self.concatenate_wave_functions_3_phases(V_0_pool_f1, V_0_pool_f2, V_0_pool_b1, V_0_pool_b2, 
                                                                                                  last_p1, first_p2, last_p2, first_p3)

        energy_object = energy.energy(Mx=self.Mx, My=self.My, B=self.B, V_0=0, tx=self.tx, ty=self.ty,
                                      qx=self.qx, qy=self.qy, n=self.n, x=self.x, dt=self.dt, tol=self.tol)

        E_col_1 = np.zeros(len(V1))
        E_col_2 = np.zeros(len(V2))
        E_col_3 = np.zeros(len(V3))

        for i in range(len(V1)):
            energy_object.V_0 = V1[i]
            E_col_1[i] = energy_object.calc_energy(psi_phase1[i])[0]
        for i in range(len(V2)):
            energy_object.V_0 = V2[i]
            E_col_2[i] = energy_object.calc_energy(psi_phase2[i])[0]
        for i in range(len(V3)):
            energy_object.V_0 = V3[i]
            E_col_3[i] = energy_object.calc_energy(psi_phase3[i])[0]

        j = 0
        for i in range(int(np.where(V1 == V2[0])[0][0]), len(V1)-1):
            if E_col_2[j] < E_col_1[i]:
                last_index_V1  = i
                first_index_V2 = j
                break
            j += 1

        j = 0
        for i in range(int(np.where(V2 == V3[0])[0][0]), len(V2)-1): #+1
            if E_col_3[j] < E_col_2[i]:
                last_index_V2  = i
                first_index_V3 = j
                break
            j += 1

        V1 = V1[0:last_index_V1]
        V2 = V2[first_index_V2:last_index_V2]
        V3 = V3[first_index_V3:len(V3)]

        psi_phase1 = psi_phase1[0:last_index_V1]
        psi_phase2 = psi_phase2[first_index_V2:last_index_V2]
        psi_phase3 = psi_phase3[first_index_V3:len(psi_phase3)]
    
        return V1.copy(), V2.copy(), V3.copy(), psi_phase1.copy(), psi_phase2.copy(), psi_phase3.copy()
