import os, sys, json

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp

import h5py

###########################################################################################
############## MAIN #######################################################################
###########################################################################################
def main():
    io_class = input_output()
    params = io_class.get_parameters(1)

    # param objects
    wfnpar = WfnParams(int(params["Mx"]), int(params["My"]), int(params["n"]))
    hamiltpar = HamiltParams(float(params["B"]), float(params["tx"]), float(params["ty"]), float(params["qx"]), float(params["qy"]), float(params["V_0"]))

    # wavefunc objects
    wfn_manip = wavefunc_operations(params=params)
    wfn_init = wavefunctions(params=params)

    # initialize wavefunction
    psi_init = wfn_init.create_init_wavefunction(params["init"])
    psi_init = wfn_manip.reshape_one_dim(psi_init)

    # energy objects
    energy_object = energy(params=params)
    eom_object = eom(params=params)

    epsilon = 1
    tol = params['tol']
    tol = float(params["tol"])
    rtol = float(params["rtol"]) 
    atol = float(params["atol"]) 
    dtau = float(params["dtau"])
    
    iter = 0
    while epsilon > tol:
        '''
        imag time evolution for dt
        '''
        sol = solve_ivp(lambda t_, psi_ : -1.0*eom_object.rhs_lang_firsov_imag_time_prop(psi_.reshape((wfnpar.My,wfnpar.Mx,wfnpar.n))),
                         [0,dtau], psi_init, method='RK45', rtol=rtol, atol=atol) # method options: 'RK45', 'DOP853'

        '''
        normalize
        '''
        psi_iter = sol.y.T[-1]
        psi_iter = wfn_manip.normalize_wf(psi_iter, shape=(wfnpar.Mx*wfnpar.My,wfnpar.n))

        '''
        energy and epsilon criterion
        '''
        E = energy_object.calc_energy(psi_iter)
        epsilon = eom_object.epsilon_criterion_single_rotor(psi_iter, psi_init) # NOTE: other criteria available

        print('V_0 =', hamiltpar.V0, ', iter step = ' + str(iter+1)+", E =", E[0].real, ", epsilon =", epsilon)

        '''
        update psi_init
        '''
        psi_init = wfn_manip.reshape_one_dim(psi_iter)
        iter = iter + 1

    psi_out = wfn_manip.reshape_three_dim(psi_init)

    with h5py.File(f'data_vgh/data_{params["init"]}_B_{hamiltpar.B}_V0_{hamiltpar.V0}_tx_{hamiltpar.tx}_ty_{hamiltpar.ty}_qx_{hamiltpar.qx}_qy_{hamiltpar.qy}_{wfnpar.Mx}x{wfnpar.My}.h5', 'w') as f:
        f.create_dataset('Mx', data=wfnpar.Mx)
        f.create_dataset('My', data=wfnpar.My)
        f.create_dataset('n', data=wfnpar.n)
        f.create_dataset('B', data=hamiltpar.B)
        f.create_dataset('tx', data=hamiltpar.tx)
        f.create_dataset('ty', data=hamiltpar.ty)
        f.create_dataset('qx', data=hamiltpar.qx)
        f.create_dataset('qy', data=hamiltpar.qy)
        f.create_dataset('V0', data=hamiltpar.V0)
        f.create_dataset('wfn', data=psi_out)
        

###########################################################################################
############## CLASSES ####################################################################
###########################################################################################
class WfnParams:

    def __init__(self, Mx, My, n):
        self.Mx = Mx
        self.My = My
        self.n = n
        self.calculate_grids()

    def calculate_grids(self):
        n = self.n
        self.phi = 2*np.pi/n * np.arange(n)
        self.k2 = -np.concatenate((np.arange(n/2+1), np.arange(n/2+1, n) - n))**2

    def read(self):
        return (self.Mx, self.My, self.n, self.phi, self.k2)

class input_output:
    def get_file_path(self, arg):
        try:
            argument = sys.argv[int(arg)]
            if argument == "-h" or argument == "h" or argument == "-help" or argument == "help": print(" "); print("Use this argument structure: [PATH]"); print(" "); quit()
            else: file_path = argument
            print(" "); print("Verify Path: ", file_path); print(' ')
        except:
            print(" "); print("Verify Path: ", file_path); print(' ')
            pass
        
        return file_path

    def get_parameters(self, arg):
        path_main = os.path.dirname(os.path.abspath(__file__))+"/"
        file_path = self.get_file_path(arg)
        
        with open(path_main+file_path) as file:
            data = file.read()

        param_dict = json.loads(data)

        for key, value in param_dict.items():
            print(key,'=',value)

        return param_dict

class Wavefunction:
    '''Class to analyze numpy arrays as wavefunctions'''

    def __init__(self, params, array=None):

        if type(params) is WfnParams:
            self.par = params
            self.initialize()
        elif type(params) is Wavefunction:
            self.par = params.par
            self.initialize()
            self.wfn = params.wfn.copy()
            
        if isinstance(array, np.ndarray):
            if array.shape == self.wfn.shape:
                self.wfn[...] = array.copy()
            else:
                raise TypeError('Incorrect size of wavefunction array')

    def initialize(self):
        Mx, My, n, phi, k2 = self.par.read()

        ## create array
        self.wfn = np.zeros(((n+1)*Mx*My,), dtype=complex)
        ## create views
        self.elewfn = self.wfn[0:(Mx*My)].view().reshape((Mx, My))
        self.rotwfn = self.wfn[(Mx*My):].view().reshape((n, Mx, My))

    def initialize_localized(self, length_ele):
        Mx, My, n, phi, k2 = self.par.read()

        ## initialize views
        self.rotwfn[...] = np.full((n, Mx, My), n**(-0.5), dtype=complex)
        self.elewfn[...] = np.exp(-0.5*(np.arange(Mx)[:, np.newaxis] - Mx/2)**2/length_ele
                          -0.5*(np.arange(My)[np.newaxis, :] - My/2)**2/length_ele)
        self.normalize()

    def construct(self, elewfn, rotwfn):
        self.rotwfn[...] = rotwfn
        self.elewfn[...] = elewfn

    def elewfn_shift(self, m, n):
        return np.roll(self.elewfn, (m, n), axis=(0, 1))

    def rotwfn_shift(self, m, n):
        return np.roll(self.rotwfn, (m, n), axis=(1, 2))

    def rotwfn_d2dx(self):
        return np.fft.ifft(np.einsum('k,kij->kij', self.par.k2,
                                     np.fft.fft(self.rotwfn, axis=0)), axis=0)

    def normalize(self):
        ampl_ele = np.sum(abs(self.elewfn)**2, axis=(0,1))
        ampl_rot = np.sum(abs(self.rotwfn)**2, axis=(0))

        self.elewfn[...] = self.elewfn / np.sqrt(ampl_ele)
        self.rotwfn[...] = np.einsum('kij,ij->kij', self.rotwfn, ampl_rot**(-0.5))


class HamiltParams:

    def __init__(self, B, tx, ty, qx, qy, V0):
        self.B  = B
        self.tx = tx
        self.ty = ty
        self.qx = qx
        self.qy = qy
        self.V0 = V0
        self.std_geometry()

    def read(self):
        return (self.B, self.tx, self.ty, self.V0, self.eta)

    def std_geometry(self):
        self.eta = np.zeros((4,), dtype=float)
        self.eta[0] = -np.pi/4.0
        self.eta[1] = -3.0*np.pi/4.0
        self.eta[2] = np.pi/4.0
        self.eta[3] = 3.0*np.pi/4.0

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

    def calc_energy_sym_breaking(self, psi):
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

        angle_pattern = np.array(self.param_dict['angle_pattern'])
        #V_0_pattern = np.array(self.param_dict['V_0_pattern'])
        
        '''
        interaction energy
        '''
        E_V = self.V_0*np.sum(np.cos(self.x-0.25*np.pi+angle_pattern[0])*np.abs(psi[self.My-1,0])**2)
        E_V += self.V_0*np.sum(np.cos(self.x-0.75*np.pi+angle_pattern[1])*np.abs(psi[self.My-1,self.Mx-1])**2)
        E_V += self.V_0*np.sum(np.cos(self.x+0.25*np.pi+angle_pattern[2])*np.abs(psi[0,0])**2)
        E_V += self.V_0*np.sum(np.cos(self.x+0.75*np.pi+angle_pattern[3])*np.abs(psi[0,self.Mx-1])**2)
        
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
        wfn2_manip = permute_rotors(psi2c)

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
        wfn2_manip = permute_rotors(psi2)

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

class eom:
    ''' Class for evaluation of the equations of motion for a rotor lattice ...

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

            dt (float): time step of one Runge-Kutta propagation
            time_steps (int): number of time steps in the real time propagation
            tol (float): convergence criterion for the ground state in the imaginary time propagation
        ----

        but most importantly:

        ----
        Methods:
            self.hpsi_lang_firsov(wavefunc. as three-dimensional numpy array): calculate H_psi
                                                        of the variational equation of motion
            self.rhs_lang_firsov_imag_time_prop(wavefunc. as three-dimensional numpy array):
            self.rhs_lang_firsov_real_time_prop(wavefunc. as three-dimensional numpy array):
        ----
    '''    

    def __init__(self, params):
        self.param_dict = params
        self.Mx  = int(params['Mx'])
        self.My  = int(params['My'])
        self.M   = int(params['Mx']*params['My'])
        self.B   = float(params['B'])
        self.V_0 = 0 if isinstance(params['V_0'], list) == True else float(params['V_0'])
        self.tx  = float(params['tx'])
        self.ty  = float(params['ty'])
        self.qx  = int(params['qx'])
        self.qy  = int(params['qy'])
        self.n   = int(params['n'])
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid

    def compute_T_matrix_eom(self, Tarr):
        '''
            Computes: the transfer integrals in the equations of motion

            ----
            Inputs:
                Tarr (2-dimensional: (My,Mx)): contains the overlaps of the single rotor wavefunctions
            ----

            ----
            Outputs:
                Tarr_new (3-dimensional: (My,Mx,n)): the muliplied transfer integrals except the (i,j) one
            ----
        '''
        T_cur_arr = Tarr.copy()
        Tarr_new = np.zeros(Tarr.shape, dtype=complex)

        for i in range(self.My):
            for j in range(self.Mx):
                T_cur_arr[i,j] = 1.0+0j
                Tarr_new[i,j] = np.prod(T_cur_arr)

                T_cur_arr[i,j] = Tarr[i,j]

        Tarr_new = Tarr_new[:, :, np.newaxis]
        return Tarr_new

    def hpsi_transfer_and_rot(self, psi_collection):
        '''
            Computes: The transfer and rotational part of Hpsi

            ----
            Inputs:
                psi_collection (3-dimensional: (My, Mx, n), dtype: complex): stores the rotor wavefunctions
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

                k2: second derivative matrix
            ----

            ----
            Outputs:
                H_psi (2-dimensional: (My*Mx,n)): H_psi of the variational equations of motion
            ----
        '''
        
        # object for manipulating wavefunctions
        '''
            TODO: think about combining the kinetic energy calculation with the class_energy object!!
        '''
        wfn_manip = permute_rotors(psi_collection)

        psi_collection_conj = np.conjugate(psi_collection)

        '''
        arrays of transfer integrals: all are (My,Mx) objects
        '''
        TD_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_next_y_rotor(), dtype=complex) 
        TU_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_prev_y_rotor(), dtype=complex)
        TR_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_next_x_rotor(), dtype=complex)
        TL_arr = np.einsum('ijk,ijk->ij', psi_collection_conj, wfn_manip.get_prev_x_rotor(), dtype=complex)

        '''
        products of transfer integrals: in every entry (i,j), the (i,j)-th produc is missing
        '''
        TDr = self.compute_T_matrix_eom(TD_arr)
        TUr = self.compute_T_matrix_eom(TU_arr)
        TRr = self.compute_T_matrix_eom(TR_arr)
        TLr = self.compute_T_matrix_eom(TL_arr)
        
        '''
        rotor kinetic energy
        '''
        k2  = -np.append(np.arange(0,self.n/2+1),np.arange(-self.n/2+1,0))**2 # make second derivative matrix
        H_psi = -self.B*np.fft.ifft(k2*np.fft.fft(psi_collection)).astype(complex)
        
        '''
        electron kinetic energy
        '''
        H_psi -= self.ty*( \
                    np.exp(-1j*(2*np.pi*self.qy/self.My))*TDr*wfn_manip.get_next_y_rotor() \
                    + np.exp(+1j*(2*np.pi*self.qy/self.My))*TUr*wfn_manip.get_prev_y_rotor()) 
        H_psi -= self.tx*( \
                    np.exp(-1j*(2*np.pi*self.qx/self.Mx))*TRr*wfn_manip.get_next_x_rotor() \
                    + np.exp(+1j*(2*np.pi*self.qx/self.Mx))*TLr*wfn_manip.get_prev_x_rotor())

        return H_psi

    def hpsi_lang_firsov(self, psi_collection):
        '''
            Computes: H_psi of the variational equations of motion

            ----
            Inputs:
                psi_collection (3-dimensional: (My, Mx, n), dtype: complex): stores the rotor wavefunctions
            ----

            ----
            Outputs:
                H_psi (2-dimensional: (My*Mx,n)): H_psi of the variational equations of motion
            ----
        '''
        
        H_psi = self.hpsi_transfer_and_rot(psi_collection)
        
        '''
        interaction terms
        '''
        H_psi[self.My-1,0]         += self.V_0*np.cos(self.x-0.25*np.pi)*psi_collection[self.My-1,0]
        H_psi[self.My-1,self.Mx-1] += self.V_0*np.cos(self.x-0.75*np.pi)*psi_collection[self.My-1,self.Mx-1]
        H_psi[0,0]                 += self.V_0*np.cos(self.x+0.25*np.pi)*psi_collection[0,0]
        H_psi[0,self.Mx-1]         += self.V_0*np.cos(self.x+0.75*np.pi)*psi_collection[0,self.Mx-1]

        return H_psi

    def rhs_lang_firsov_imag_time_prop(self, psi_collection):
        '''
            Computes: right-hand-side of the variational e.o.m. for imaginary time propagation

            ----
            Inputs:
                psi_collection (3-dimensional: (My, Mx, n), dtype: complex): stores the rotor wavefunctions
            ----

            ----
            Variables: 
                H_psi (2-dimensional: (My*Mx,n)): right-hand-side of the variational equations of motion
                lagrange_param (2-dimensional: (My*Mx,n)): lagrange parameter to ensure normalization
            ----

            ----
            Outputs:
                H_psi (1-dimensional: (My*Mx*n)): right-hand-side of the variational equations of motion for imag time evolution
            ----
        '''
        H_psi = self.hpsi_lang_firsov(psi_collection)

        lagrange_multiplier = np.einsum('ijk,ijk->ij', np.conjugate(psi_collection), H_psi)
        
        H_psi = H_psi - lagrange_multiplier[:, :, np.newaxis] * psi_collection
        H_psi = H_psi.flatten()

        return H_psi
    
    def epsilon_criterion_rhs_prev_rhs_next(self, psi_t1, psi_t2):
        '''
            Computes: epsilon criterion based on respective overlaps of rhs of psi_t1 and psi_t2

            ----
            Inputs:
                psi_t1 (3-dimensional: (My,Mx,n)): wavefunction at time step t-1 
                psi_t2 (3-dimensional: (My,Mx,n)): wavefunction at time step t
            ----

            ----
            Variables:
                rhs_t1 (1-dimensional: (M*n)): rhs of e.o.m. for psi at t-1
                rhs_t2 (1-dimensional: (M*n)): rhs of e.o.m. for psi at t
            ----

            ----
            Outputs:
                epsilon (scalar, real): between 0 and 1, corresponds to variance of energy (zero for GS)
            ----
        '''
        
        psi_t1 = psi_t1.reshape((self.My,self.Mx,self.n))
        psi_t2 = psi_t2.reshape((self.My,self.Mx,self.n))

        rhs_t1 = self.rhs_lang_firsov_imag_time_prop(psi_t1)
        rhs_t2 = self.rhs_lang_firsov_imag_time_prop(psi_t2)

        epsilon = np.abs(np.sum(np.conjugate(rhs_t1)*rhs_t2))
        return epsilon
    
    def epsilon_criterion_rhs(self, psi_t1):
        '''
            Computes: epsilon criterion based on variance of energy

            ----
            Inputs:
                psi_t1 (3-dimensional: (My,Mx,n)): wavefunction at time step t-1 
            ----

            ----
            Variables:
                rhs_t1 (1-dimensional: (M*n)): rhs of e.o.m. for psi at t
            ----

            ----
            Outputs:
                epsilon (scalar, real): between 0 and 1, corresponds to variance of energy (zero for GS)
            ----
        '''
        
        psi_t1 = psi_t1.reshape((self.My,self.Mx,self.n))

        rhs_t1 = self.rhs_lang_firsov_imag_time_prop(psi_t1)

        epsilon = np.abs(np.sum(np.conjugate(rhs_t1)*rhs_t1))
        return epsilon
    
    def epsilon_criterion_single_rotor(self, psi_t1, psi_t2):
        '''
            Computes: epsilon criterion based on minimal single rotor overlap

            ----
            Inputs:
                psi_t1 (3-dimensional: (My,Mx,n)): wavefunction at time step t-1 
                psi_t2 (3-dimensional: (My,Mx,n)): wavefunction at time step t
            ----

            ----
            Variables:
                wfn_manip: object to handle wavefunction
                single_rotor_overlap (1-dimensional (M,)): overlap of single rotors of psi_t1 and psi_t2
            ----

            ----
            Outputs:
                epsilon (scalar, real): between 0 and 1
            ----
        '''
        wfn_manip = wavefunc_operations(params=self.param_dict)
        single_rotor_overlap = wfn_manip.single_rotor_overlap(psi_t1, psi_t2)
        epsilon = 1 - np.abs(np.min(single_rotor_overlap))
        
        return epsilon

class wavefunctions:
    ''' Class for wavefunction creation
        ----
        
        ----
        Inputs: 
            params: dictionary that contains the class variables
        ----

        ----
        Class variables:
            n (int): length of angle grid
            Mx (int): number of rotors in x direction
            My (int): number of rotors in y direction
            M (int): Mx*My, total number of rotor
        ----

        ----
        Methods:
            self.create_init_wavefunction(phase): computes "approximate" psi (My,Mx,n) with symmetry specified by input variable 'phase'
        ----
    '''

    def __init__(self, params):
        self.param_dict = params
        self.Mx  = int(params['Mx'])
        self.My  = int(params['My'])
        self.M   = int(params['Mx']*params['My'])
        self.B   = float(params['B'])
        self.V_0 = 0 if isinstance(params['V_0'], list) == True else float(params['V_0'])
        self.tx  = float(params['tx'])
        self.ty  = float(params['ty'])
        self.qx  = int(params['qx'])
        self.qy  = int(params['qy'])
        self.n   = int(params['n'])
        '''
            TODO: make external functional that creates the angle grid - such that this can be changed externally somehow!
        '''
        self.x   = (2*np.pi/self.n)*np.arange(self.n) # make phi (=angle) grid
    
    def init_uniform(self):
        '''
            Computes: uniform wavefunction

            ----
            Inputs: 
                None
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): uniform wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output uniform wavefunction
            ----
        '''

        psi_init = self.n**(-0.5)*np.ones((self.My,self.Mx,self.n),dtype=complex)
        return psi_init
    
    def init_ferro_domain(self, orientation):
        '''
            Computes: ferroelectric domain wall wavefunction

            ----
            Comment:
                the parametrization of the wavefunction is empirical!
            ----

            ----
            Inputs: 
                orientation (string): OPTIONS: 'vertical' or 'horizontal', i.e. orientation of domain wall
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): ferroelectric domain wall wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction
            ----
        '''

        psi_init = self.init_uniform() # create object

        sigma_gauss = 0.5
        norm2 = np.sum(np.abs(np.exp(-(1-np.cos(self.x))/sigma_gauss**2)**2))

        if orientation == 'vertical':
            for i in range(self.My):
                # other option is to try with np.cos(0.5*x) and np.sin(...)

                # left column
                psi_init[i,self.Mx-1] = (1/norm2**0.5)*np.exp(-(1-np.cos(self.x))/sigma_gauss**2)  #np.cos(self.x) + 0j
                #psi_init[i,self.Mx-1][int(self.n/4):int(self.n/2)] = 0.01 + 0j
                #psi_init[i,self.Mx-1][int(self.n/2):int(3*self.n/4)] = 0.01 + 0j

                # right column
                psi_init[i,0] = (1/norm2**0.5)*np.exp(-(1-np.cos(self.x-np.pi))/sigma_gauss**2) #np.cos(self.x) + 0j
                #psi_init[i,0][0:int(self.n/4)] = 0.01 + 0j
                #psi_init[i,0][int(3*self.n/4):self.n] = 0.01 + 0j
                
        elif orientation == 'horizontal':
            for j in range(self.Mx):
                # top row
                psi_init[self.My-1,j] = (1/norm2**0.5)*np.exp(-(1-np.cos(self.x-3*np.pi/2))/sigma_gauss**2) #np.sin(self.x) 
                #psi_init[self.My-1,j][0:int(self.n/2)] = 0.01 

                # bottom row
                psi_init[0,j] = (1/norm2**0.5)*np.exp(-(1-np.cos(self.x-np.pi/2))/sigma_gauss**2) #np.sin(self.x) 
                #psi_init[0,j][int(self.n/2):self.n] = 0.01 
        return psi_init
    
    def init_small_polaron(self):
        '''
            Computes: analytic small polaron wavefunction

            ----
            Comments: 
                Here we generate the GS Mathieu functions with even n=0, although it would be possible to generate excited Mathieu states (ask Georgios)
                TODO: discuss whether we should consider excited Mathieu states?
            ----

            ----
            Inputs: 
                None
            ----
            
            ----
            Variables:
                mathieu_parameter (scalar): parameter of the mathieu equation
                y (dimension: n): mathieu function for the parameter mathieu_parameter and on x-axis
                yp (dimension: n): first derivative of mathieu function
                psi_init (3-dimensional: My,Mx,n): small polaron wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction
            ----
        '''

        mathieu_parameter = 2*self.V_0/self.B
        psi_init = self.init_uniform() # create object

        # bottom left
        y, yp = scipy.special.mathieu_cem(0, mathieu_parameter, (self.x+3*np.pi/4)/2*180/np.pi)
        psi_init[0,self.Mx-1] = y/np.sqrt(np.sum(y*y))
        
        # bottom right
        y, yp = scipy.special.mathieu_cem(0, mathieu_parameter, (self.x+np.pi/4)/2*180/np.pi)
        psi_init[0,0] = y/np.sqrt(np.sum(y*y))

        # top left
        y, yp = scipy.special.mathieu_cem(0, mathieu_parameter, (self.x-3*np.pi/4)/2*180/np.pi)
        psi_init[self.My-1,self.Mx-1] = y/np.sqrt(np.sum(y*y))

        # top right
        y, yp = scipy.special.mathieu_cem(0, mathieu_parameter, (self.x-np.pi/4)/2*180/np.pi)
        psi_init[self.My-1,0] = y/np.sqrt(np.sum(y*y))

        return psi_init
    
    def init_random(self):
        '''
            Computes: smooth random wavefunction

            ----
            Inputs: 
                None
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): small polaron wavefunction
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction
            ----
        '''

        psi_init = self.init_uniform # create object

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

        return psi_init
    
    def create_init_wavefunction(self, phase):
        '''
            Computes: initial wavefunctions, mainly for imag time propagation

            ----
            Inputs:
                phase (string): specified in input file, options:
                    - phase == 'uniform': Y_11
                    - phase == 'ferro_domain_vertical_wall': polarized domain wall states, vertically
                    - phase == 'ferro_domain_horizontal_wall': polarized domain wall states, horizontally
                    - phase == 'random': continous random wavefunctions
                    - phase == 'small_polaron': analytic Mathieu function for the 4 inner rotors
                    - phase == 'external': initialize with an external wavefunction 
            ----
            
            ----
            Variables:
                psi_init (3-dimensional: My,Mx,n): array which functions are to be defined here
            ----

            ----
            Outputs:
                psi_init (3-dimensional: (My,Mx,n)): output wavefunction with the initialized symmetry
            ----
        '''

        if phase == 'uniform':
            psi_init = self.init_uniform()

        elif phase == 'fdv': 
            psi_init = self.init_ferro_domain('vertical')
                
        elif phase == 'fdh': 
            psi_init = self.init_ferro_domain('horizontal')

        elif phase == 'sp': 
            psi_init = self.init_small_polaron()

        elif phase == 'random': 
            psi_init = self.init_random()

        elif phase == 'external': 
            '''
                TODO: implement check whether the sizes are correct
            '''
            path_to_file = self.param_dict['path_to_input_wavefunction']
            psi_init = self.init_read_in_wf(path_to_file)

        else: # sanity check
            return 
        
        wfn_manip = wavefunc_operations(params=self.param_dict)
        psi_init = wfn_manip.normalize_wf(psi_init, shape=(self.My,self.Mx,self.n))
        return psi_init

class wavefunc_operations:
    ''' Class for elementary wavefunc operations
        ----
        
        ----
        Inputs: 
            params: dictionary that contains the class variables
        ----

        ----
        Class variables:
            n (int): length of angle grid
            Mx (int): number of rotors in x direction
            My (int): number of rotors in y direction
            M (int): Mx*My, total number of rotor
        ----

        ----
        Methods:
            self.normalization_factor_wf(psi): computes 1/norm = normalization factor for every individual rotor
            self.normalize_wf(psi, shape): outputs a w.f. in which every individual rotor is normalized
            self.reshape_one_dim(psi): reshapes psi to My*Mx*n
            self.reshape_two_dim(psi): reshapes psi to (My*Mx,n)
            self.reshape_three_dim(psi): reshapes psi to (My,Mx,n)
        ----
    '''    

    def __init__(self, params):
        self.param_dict = params
        self.Mx = int(params['Mx'])
        self.My = int(params['My'])
        self.M  = int(params['Mx']*params['My'])
        self.n  = int(params['n'])

    def normalization_factor_wf(self, psi):
        '''
            ----
            Description: computes the norm factor for every rotor
                (1) take abs(psi)**2 of every rotor
                (2) sum over the 2nd axis (axis=1), i.e. sum over angles
                (3) take the square and the inverse to get the norm-factor
                (4) this gives a (M,1) array, specifying the normalization factor for every rotor
            ----

            ----
            Inputs:
                psi (shape doesn't matter: max. 3-dimensional)
            ----
            
            ----
            Variables:
                norm2 (2-dimensional: (M,n)): norm2 of psi
                norm_sqrt (2-dimensional: (M,1)): sqrt of norm2, summed over angle n, i.e. sqrt of norm for every single rotor 
            ---- 
            
            ----
            Outputs:
                normalization_factor (shape: (My*Mx,1)): normalization factor for every rotor
            ----
        '''
        psi = self.reshape_two_dim(psi) 

        norm2 = np.abs(psi)**2
        norm_sqrt = np.sqrt(np.sum(norm2,axis=1)).reshape(self.M,1)

        normalization_factor = 1.0/norm_sqrt
        return normalization_factor
    
    def normalize_wf(self, psi, shape):
        ''' 
            ----
            Description: normalizes every single rotor
            ----

            ----
            Inputs:
                psi (shape doesn't matter: max. 3-dimensional): wavefunction to normalize
            ----

            ----
            Outputs:
                psi (shape as specified by input shape=(,,)): normalized wavefunction
            ----
        '''

        normalization_factor = self.normalization_factor_wf(psi) # 1./norm 
        psi = normalization_factor*self.reshape_two_dim(psi)
        return psi.reshape(shape)
    
    def reshape_one_dim(self, psi):
        '''
            ----
            Output:
                psi (1-dimensional (My*Mx*n))
            ----
        '''
        return psi.reshape((self.My*self.Mx*self.n))
    
    def reshape_two_dim(self, psi):
        '''
            ----
            Output:
                psi (2-dimensional (My*Mx,n))
            ----
        '''
        return psi.reshape((self.My*self.Mx,self.n))
    
    def reshape_three_dim(self, psi):
        '''
            ----
            Output:
                psi (3-dimensional (My,Mx,n))
            ----
        '''
        return psi.reshape((self.My,self.Mx,self.n))

    def calc_overlap(self, psi1, psi2):
        ''' 
            ----
            Description: total overlap of psi1 and psi2
            ----

            ----
            Inputs:
                psi1 (max. 3-dimensional, but dimension is checked)
                psi2 (max. 3-dimensional, but dimension is checked)
            ----

            ----
            Variables:
                psi1_conj: conjugate of psi1
                overlap (scalar, dtype=complex): total overlap
            ----
            Output:
                overlap
            ----
        '''

        psi1 = self.reshape_three_dim(psi1) # for safety, to ensure that it is always of same shape
        psi2 = self.reshape_three_dim(psi2) # psi2.reshape((self.My, self.Mx, self.n)) # for safety, to ensure that it is always of same shape

        psi1_conj = np.conjugate(psi1)

        overlap = 1 + 0j
        for k in range(self.My): 
            for p in range(self.Mx):
                overlap *= np.sum(psi1_conj[k,p]*psi2[k,p])
        return overlap
    
    def single_rotor_overlap(self, psi1, psi2):
        ''' 
            ----
            Description: single rotor overlaps of psi1 and psi2 
            ----

            ----
            Inputs:
                psi1 (max. 3-dimensional, but dimension is checked)
                psi2 (max. 3-dimensional, but dimension is checked)
            ----

            ----
            Variables:
                psi1_conj: conjugate of psi1
            ----

            ----
            Output:
                overlap (1-dimensional (M,)): overlap of the M rotors in psi1 and psi2 
            ----
        '''

        psi1 = self.reshape_two_dim(psi1) # now a (M,n) object
        psi2 = self.reshape_two_dim(psi2) # now a (M,n) object

        psi1_conj = np.conjugate(psi1)

        '''
        sum over angle axis
        '''
        overlap = np.sum(psi1_conj*psi2, axis=1) 
        return overlap
    
    def cut_out_rotor_region(self, psi, chosen_My, chosen_Mx):
        ''' 
            ----
            Description: 
                - Cuts out a specified number of rotors, given by chosen_My and chosen_Mx
                - Is the contrast to the function add_rotors_to_wavefunction(...)
            ----

            ----
            Inputs:
                psi (3-dimensional: (My,Mx,n)): input wavefunction
                chosen_My (int, scalar): the chosen number of y rotors
                chosen_Mx (int, scalar): the chosen number of x rotors
            ----

            ----
            Outputs:
                psi_new (3-dimensional: (chosen_My, chosen_Mx, n)): wavefunction with smaller number of rotors
            ----
        '''

        psi_new = np.zeros((chosen_My,chosen_Mx,self.n), dtype=complex)

        for i in range(self.My):
            for j in range(self.Mx):
                border_i_left  = int((self.My-chosen_My)/2)
                border_i_right = int((self.My+chosen_My)/2)

                border_j_left  = int((self.Mx-chosen_Mx)/2)
                border_j_right = int((self.Mx+chosen_Mx)/2)

                if i >= border_i_left and i < border_i_right:
                    if j >= border_j_left and j < border_j_right:
                        psi_ind_i = (i+int(self.My/2))%self.My
                        psi_ind_j = (j+int(self.Mx/2))%self.Mx

                        psi_new[i-border_i_left, j-border_j_left] = psi[psi_ind_i,psi_ind_j]
                        
        return psi_new
    
    def individual_rotor_density(self, psi, chosen_My, chosen_Mx):
        ''' 
            ----
            Description: computes the density for every single rotor
            ----

            ----
            Inputs:
                psi (3-dimensional: (chosen_My,chosen_Mx,n)): input wavefunction
                chosen_My (int, scalar): the chosen number of y rotors
                chosen_Mx (int, scalar): the chosen number of x rotors
            ----

            ----
            Outputs:
                rotor_density (3-dimensional: (chosen_My, chosen_Mx, n)): rotor density for every individual rotor
            ----
        '''

        rotor_density = np.zeros((chosen_My,chosen_Mx,self.n), dtype=complex)

        for i in range(chosen_My):
            for j in range(chosen_Mx):
                #psi_ind_i = (i+int(chosen_My/2))%chosen_My
                #psi_ind_j = (j+int(chosen_Mx/2))%chosen_Mx

                ind_rotor_psi = psi[i,j]
                rotor_density[i,j] = (np.conjugate(ind_rotor_psi)*ind_rotor_psi).T 

        return rotor_density
    
    def individual_rotor_phase(self, psi, chosen_My, chosen_Mx):
        ''' 
            ----
            Description: computes the phase for every single rotor
            ----

            ----
            Comment: there is another way to compute the phase, instead of using the numpy function:
                sign_fac = np.sign(ind_rotor_psi.imag) # an (n,) object
                phase_without_sign = np.arccos(ind_rotor_psi.real/np.abs(ind_rotor_psi)) # an (n,) object
                
                phase = sign_fac*phase_without_sign
            ----
            Inputs:
                psi (3-dimensional: (chosen_My,chosen_Mx,n)): input wavefunction
                chosen_My (int, scalar): the chosen number of y rotors
                chosen_Mx (int, scalar): the chosen number of x rotors
            ----

            ----
            Outputs:
                rotor_pase (3-dimensional: (chosen_My, chosen_Mx, n)): rotor density for every individual rotor
            ----
        '''
        
        rotor_pase = np.zeros((chosen_My,chosen_Mx,self.n), dtype=complex)

        for i in range(chosen_My):
            for j in range(chosen_Mx):
                #psi_ind_i = (i+int(chosen_My/2))%chosen_My
                #psi_ind_j = (j+int(chosen_Mx/2))%chosen_Mx

                ind_rotor_psi = psi[i,j]

                phase = np.arctan2(ind_rotor_psi.imag,ind_rotor_psi.real) 

                rotor_pase[i,j] = phase

        return rotor_pase

class permute_rotors:
    ''' Class for moving the rotors in the different directions
        ----
        
        ----
        Inputs: 
            psi (3-dimensional: (My,Mx,n)): input psi
        ----

        ----
        Outputs:
            psi (3-dimensional: (My,Mx,n)): but here, one column or row was shifted
        ----

        ----
        Methods: 
            (Note for below: convention in array: [My,Mx,n]; [i,j] means picking the (i-th,j-th) rotor from the array)
            self.get_next_y_rotor(): "equivalent" to [(i+1)%self.My,j]
            self.get_prev_y_rotor(): "equivalent" to [i-1,j]
            self.get_next_x_rotor(): "equivalent" to [i,(j+1)%self.Mx]
            self.get_prev_x_rotor(): "equivalent" to [i,j-1]
        ----
    '''

    def __init__(self, psi):
        self.psi = psi

    def get_next_y_rotor(self):
        return np.roll(self.psi, -1, axis=0)
    
    def get_prev_y_rotor(self):
        return np.roll(self.psi, 1, axis=0)
    
    def get_next_x_rotor(self):
        return np.roll(self.psi, -1, axis=1)
    
    def get_prev_x_rotor(self):
        return np.roll(self.psi, 1, axis=1)
    
if __name__ == '__main__':
    main()
