import os, sys, json

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

###########################################################################################
############## MAIN #######################################################################
###########################################################################################
def main():
    io_class = input_output()
    params = io_class.get_parameters(0)

    #wfnpar = WfnParams(10, 10, 256)
    #hamiltpar = HamiltParams(0.5e-2, 0.5, 0.5, 0.3)
    #n, Mx, My, V_0, B, tx, ty, tol, rtol, atol, dtau, init_gauss

    wfnpar = WfnParams(params["Mx"], params["My"], params["n"])
    hamiltpar = HamiltParams(params["B"], params["tx"], params["ty"], params["V_0"])

    wfn = Wavefunction(wfnpar)
    wfn.initialize_localized(params["init_gauss"])

    H = Hamilt(wfnpar, hamiltpar)

    epsilon = 1
    tol = params["tol"] #4.0e-5
    rtol = params["rtol"] #1e-9
    atol = params["atol"] #1e-9
    dtau = params["dtau"] #1.0

    energy_old, energy_rot, energy_ele, energy_int = H.calculate_energy(wfn)
    # plot1 = DynamicPlot(H, wfn)
    while epsilon > tol:
        obj = solve_ivp(lambda tau, array: H.apply(array),
                        (0, dtau), wfn.wfn, method='RK45',
                        t_eval=[dtau], rtol=rtol, atol=atol)

        wfn_new = Wavefunction(wfnpar, array=obj.y[:, 0])
        wfn_new.normalize()

        # convergence condition (here rhs zero)
        # epsilon = np.sum(np.abs(H.apply(wfn_new.wfn)))

        energy_tot, energy_rot, energy_ele, energy_int = H.calculate_energy(wfn_new)

        # energy condition
        epsilon = abs(energy_old - energy_tot)

        print(f'epsilon:{epsilon}, energy:{energy_tot.real}')

        # plot1.update_plots(wfn_new)

        wfn = wfn_new
        energy_old = energy_tot

    with h5py.File(f'data_V0_{hamiltpar.V0}_{wfnpar.Mx}x{wfnpar.My}.h5', 'w') as f:
        f.create_dataset('Mx', data=wfnpar.Mx)
        f.create_dataset('My', data=wfnpar.My)
        f.create_dataset('n', data=wfnpar.n)
        f.create_dataset('B', data=hamiltpar.B)
        f.create_dataset('tx', data=hamiltpar.tx)
        f.create_dataset('ty', data=hamiltpar.ty)
        f.create_dataset('V0', data=hamiltpar.V0)
        f.create_dataset('wfn', data=wfn.wfn)
        


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
    
        #with open(path_main+file_path, newline='') as csvfile:
        #    spamreader = csv.reader(csvfile, delimiter=' ')
        #    for row in spamreader:
        #        identifier = row[0]
        #        value = row[1].replace(" ", "")
        #        if identifier == "n": n = int(value); print('n    =', n) # grid size of the angle
        #        elif identifier == "Mx": Mx = int(value); print('Mx   =', Mx) # number of rotors - in 2D should be a square of an (even) number
        #        elif identifier == "My": My = int(value); print('My   =', My) # number of rotors - in 2D should be a square of an (even) number
        ##        elif identifier == "V_0": V_0 = float(value); print('V_0    =', V_0) # rotational energy of rotors
        #        elif identifier == "B": B = float(value); print('B    =', B) # rotational energy of rotors
        #        elif identifier == "tx": tx = float(value); print('tx   =', tx) # tunneling along columns
        #        elif identifier == "ty": ty = float(value); print('ty   =', ty); print(' ') # tunneling along rows
        #        elif identifier == "tol": tol = float(value); print('tol  =', tol) # for convergence - 1e-7 already sufficient for most qualitative behaviour (fast), e.g. 1e-12 runs significantly longer
        #        elif identifier == "rtol": rtol = float(value); print('rtol  =', rtol)
        #        elif identifier == "atol": atol = float(value); print('atol  =', atol) 
        #        elif identifier == "dt": dt = float(value); print('dt   =', dt); print(' ') # time evolution
        #        elif identifier == "init_gauss": init_gauss = float(value); print('init =', init_gauss) # spread of gaussian

        #return n, Mx, My, V_0, B, tx, ty, tol, rtol, atol, dt, init_gauss

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

    def __init__(self, B, tx, ty, V0):
        self.B  = B
        self.tx = tx
        self.ty = ty
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

class Hamilt:

    def __init__(self, *params):
        self.hamilt_parameters_set = False
        self.wavefunction_parameters_set = False

        for param in params:
            if type(param) is WfnParams:
                self.set_wfn_params(param)
            if type(param) is HamiltParams:
                self.set_hamilt_params(param)

    # bookkeeping methods
    def set_wfn_params(self, param):
        self.wfn_params = param
        self.wavefunction_parameters_set = True

    def set_hamilt_params(self, param):
        self.hamilt_params = param
        self.hamilt_parameters_set = True

    def change_B(self, B):
        if self.hamilt_parameters_set == True:
            self.hamilt_params.B = B
        else:
            raise ValueError('Hamiltonian parameters not yet set!')

    def change_tx(self, tx):
        if self.hamilt_parameters_set == True:
            self.hamilt_params.tx = tx
        else:
            raise ValueError('Hamiltonian parameters not yet set!')

    def change_ty(self, ty):
        if self.hamilt_parameters_set == True:
            self.hamilt_params.ty = ty
        else:
            raise ValueError('Hamiltonian parameters not yet set!')

    def change_V0(self, V0):
        if self.hamilt_parameters_set == True:
            self.hamilt_params.V0 = V0
        else:
            raise ValueError('Hamiltonian parameters not yet set!')

    # actually useful methods
    def apply(self, array):
        B, tx, ty, V0, eta = self.hamilt_params.read()
        Mx, My, n, phi, k2 = self.wfn_params.read()

        wfn = Wavefunction(self.wfn_params, array=array)
        effpot_rot, effpot_ele = self.calculate_effpot(wfn)

        Hpsi_rot = -B * wfn.rotwfn_d2dx() + effpot_rot * wfn.rotwfn

        Hpsi_ele = -tx * (wfn.elewfn_shift(1, 0) + wfn.elewfn_shift(-1, 0)) \
            -ty * (wfn.elewfn_shift(0, 1) + wfn.elewfn_shift(0, -1)) \
            +effpot_ele * wfn.elewfn

        lambda_ele = np.einsum('ij,ij->', np.conjugate(wfn.elewfn), Hpsi_ele)
        lambda_rot = np.einsum('kij,kij->ij', np.conjugate(wfn.rotwfn), Hpsi_rot)

        rhs_ele = lambda_ele * wfn.elewfn - Hpsi_ele
        rhs_rot = np.einsum('ij,kij->kij', lambda_rot, wfn.rotwfn) - Hpsi_rot

        Hpsi = Wavefunction(self.wfn_params)
        Hpsi.construct(rhs_ele, rhs_rot)

        return Hpsi.wfn

    def calculate_energy(self, wfn):
        B, tx, ty, V0, eta = self.hamilt_params.read()
        Mx, My, n, phi, k2 = self.wfn_params.read()

        energy_rot = -B * np.einsum('kij,kij->ij', np.conjugate(wfn.rotwfn), wfn.rotwfn_d2dx())

        Hpsi_ele = -tx * (wfn.elewfn_shift(1, 0) + wfn.elewfn_shift(-1, 0)) \
            -ty * (wfn.elewfn_shift(0, 1) + wfn.elewfn_shift(0, -1))

        energy_ele = np.einsum('ij,ij->', np.conjugate(wfn.elewfn), Hpsi_ele)

        rc = np.abs(wfn.elewfn)**2
        effpot_rot, effpot_ele = self.calculate_effpot(wfn)
        energy_int = rc * effpot_ele

        energy_tot = np.sum(energy_rot, axis=(0,1)) + energy_ele + np.sum(energy_int, axis=(0,1))

        return (energy_tot, energy_rot, energy_ele, energy_int)

    def calculate_effpot(self, wfn):
        B, tx, ty, V0, eta = self.hamilt_params.read()
        Mx, My, n, phi, k2 = self.wfn_params.read()

        ra = np.einsum('k,ij->kij', np.cos(phi - eta[0]), np.abs(wfn.elewfn_shift(1,0))**2)
        rb = np.einsum('k,ij->kij', np.cos(phi - eta[1]), np.abs(wfn.elewfn_shift(1,1))**2)
        rc = np.einsum('k,ij->kij', np.cos(phi - eta[2]), np.abs(wfn.elewfn)**2)
        rd = np.einsum('k,ij->kij', np.cos(phi - eta[3]), np.abs(wfn.elewfn_shift(0,1))**2)

        effpot_rot = V0 * (ra + rb + rc + rd)

        ea = np.einsum('k,kij->ij', np.cos(phi - eta[0]), np.abs(wfn.rotwfn_shift(-1, 0))**2)
        eb = np.einsum('k,kij->ij', np.cos(phi - eta[1]), np.abs(wfn.rotwfn_shift(-1,-1))**2)
        ec = np.einsum('k,kij->ij', np.cos(phi - eta[2]), np.abs(wfn.rotwfn)**2)
        ed = np.einsum('k,kij->ij', np.cos(phi - eta[3]), np.abs(wfn.rotwfn_shift(0,-1))**2)

        effpot_ele = V0 * (ea + eb + ec + ed)

        return (effpot_rot, effpot_ele)

class DynamicPlot:

    def __init__(self, Hamilt, wfn):
        self.Mx = np.arange(wfn.par.Mx)
        self.My = np.arange(wfn.par.My)
        self.phi = wfn.par.phi
        self.Hamilt = Hamilt

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = \
            plt.subplots(nrows=2, ncols=2)

        Px = np.einsum('k,kij->ij', np.cos(self.phi), np.abs(wfn.rotwfn)**2)
        Py = np.einsum('k,kij->ij', np.sin(self.phi), np.abs(wfn.rotwfn)**2)
        effpot_rot, effpot_ele = Hamilt.calculate_effpot(wfn)
        
        # initialize plots
        data1, self.cax1, cbar1 = self.mypcolormesh(self.ax1, self.Mx, self.My,
                                               np.abs(wfn.elewfn)**2)
        self.ax1.set_title('density electron')

        data2, self.cax2, cbar2 = self.mypcolormesh(self.ax2, self.Mx, self.My,
                                               Px**2 + Py**2)
        self.ax2.set_title('magnitude rotor polarization')

        self.ax3.quiver(self.Mx, self.My,
                   0.5*Px/np.sqrt(Px**2 + Py**2), 0.5*Py/np.sqrt(Px**2 + Py**2))
        self.ax3.set_title('orientation rotor polarization')

        data4, self.cax4, cbar4 = self.mypcolormesh(self.ax4, self.Mx, self.My,
                                                    effpot_ele)
        self.ax4.set_title('electron effective potential')

        plt.draw()
        plt.pause(0.001)
    
    def update_plots(self, wfn):
        self.ax1.cla()
        self.cax1.cla()
        self.ax2.cla()
        self.cax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.cax4.cla()

        Px = np.einsum('k,kij->ij', np.cos(self.phi), np.abs(wfn.rotwfn)**2)
        Py = np.einsum('k,kij->ij', np.sin(self.phi), np.abs(wfn.rotwfn)**2)
        effpot_rot, effpot_ele = self.Hamilt.calculate_effpot(wfn)

        data1 = self.ax1.pcolormesh(self.Mx, self.My,
                                np.abs(wfn.elewfn)**2, rasterized=True)
        cbar1 = plt.colorbar(data1, cax=self.cax1)
        self.ax1.set_title('density electron')

        data2 = self.ax2.pcolormesh(self.Mx, self.My,
                                Px**2 + Py**2, rasterized=True)
        cbar2 = plt.colorbar(data2, cax=self.cax2)
        self.ax2.set_title('magnitude rotor polarization')

        self.ax3.quiver(self.Mx, self.My,
                   0.5*Px/np.sqrt(Px**2 + Py**2), 0.5*Py/np.sqrt(Px**2 + Py**2))
        self.ax3.set_title('orientation rotor polarization')

        data4 = self.ax4.pcolormesh(self.Mx, self.My,
                                effpot_ele, rasterized=True)
        cbar4 = plt.colorbar(data4, cax=self.cax4)
        self.ax4.set_title('electron effective potential')

        plt.draw()
        plt.pause(0.001)

    def mypcolormesh(self, axis, x, y, c):
        data = axis.pcolormesh(x, y, c, rasterized=True)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(data, cax=cax)
        return (data, cax, cbar)


if __name__ == '__main__':
    main()
