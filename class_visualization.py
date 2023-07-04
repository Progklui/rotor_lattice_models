import numpy as np
import scipy 

import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import os, sys, csv, gc

path = os.path.dirname(__file__) 
sys.path.append(path)

import class_handle_input as h_in
import class_energy as energy

# this class helps to format the axis scientifically
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

class configurations:
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
        self.dt  = float(params['dt'])
        #self.Vmin = Vmin
        #self.Vmax = Vmax

    def plot_single_rotor_density_real_time(self, rotor_density, t_index, chosen_My, chosen_Mx, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        fig, axs = plt.subplots(chosen_My,chosen_Mx, subplot_kw=dict(polar=True))
        plt.suptitle(r'$t =$'+str(t_index)+r'$\Delta t$', fontsize=font_size)

        for i in range(chosen_My):
            for j in range(chosen_Mx):
                axs[i, j].plot(self.x, rotor_density[i,j].real)

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_theta_zero_location('E')

        in_object_g = h_in.real_time(params=self.param_dict)
        folder_name_plot_g, file_name_plot_green = in_object_g.plot_rotor_density_folder_structure_real_time_prop(path_main, t_index)

        plt.savefig(folder_name_plot_g+file_name_plot_green+'.png', dpi=100)        
        plt.close()
        return 
    
    def plot_single_rotor_density_imag_time(self, rotor_density, V_0, chosen_My, chosen_Mx, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        ##plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        fig, axs = plt.subplots(chosen_My,chosen_Mx, subplot_kw=dict(polar=True))
        plt.suptitle(r'$V_0 =$'+str(V_0), fontsize=font_size)

        for i in range(chosen_My):
            for j in range(chosen_Mx):
                axs[i, j].plot(self.x, rotor_density[i,j].real)

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_theta_zero_location('E')

        in_object_g = h_in.imag_time(params=self.param_dict)
        folder_name_plot, file_name_plot = in_object_g.plot_rotor_density_folder_structure_imag_time_prop(path_main)

        plt.savefig(folder_name_plot+file_name_plot+str(V_0)+'.png', dpi=100)        
        plt.close()
        return 
    
    def plot_single_rotor_phase_real_time(self, rotor_phase, t_index, chosen_My, chosen_Mx, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        ##plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        fig, axs = plt.subplots(chosen_My,chosen_Mx, subplot_kw=dict(polar=True))
        plt.suptitle(r'$t =$'+str(t_index)+r'$\Delta t$', fontsize=font_size)

        for i in range(chosen_My):
            for j in range(chosen_Mx):
                axs[i, j].plot(self.x, rotor_phase[i,j].real)

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_theta_zero_location('E')

        in_object_g = h_in.real_time(params=self.param_dict)
        folder_name_plot_g, file_name_plot_green = in_object_g.plot_rotor_phase_folder_structure_real_time_prop(path_main, t_index)

        plt.savefig(folder_name_plot_g+file_name_plot_green+'.png', dpi=100)        
        plt.close()
        return 

    def plot_single_rotor_phase_imag_time(self, rotor_phase, V_0, chosen_My, chosen_Mx, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        ##plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        fig, axs = plt.subplots(chosen_My,chosen_Mx, subplot_kw=dict(polar=True))
        plt.suptitle(r'$V_0 =$'+str(V_0), fontsize=font_size)

        for i in range(chosen_My):
            for j in range(chosen_Mx):
                axs[i, j].plot(self.x, rotor_phase[i,j].real)

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_theta_zero_location('E')

        in_object_g = h_in.imag_time(params=self.param_dict)
        folder_name_plot, file_name_plot = in_object_g.plot_rotor_phase_folder_structure_imag_time_prop(path_main)

        plt.savefig(folder_name_plot+file_name_plot+str(V_0)+'.png', dpi=100)        
        plt.close()
        return 
    
    # plot polaron size for a specific potential - mesh on the grid
    def plot_polaron_size_real_time(self, sigma, t_index, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        M = int(self.Mx*self.My)

        fig = plt.figure()
        plt.title(r'$t =$'+str(t_index)+r'$\Delta t$', fontsize=font_size)

        pc = plt.pcolormesh(sigma, vmin=0, vmax=1)
        cbar = fig.colorbar(pc)
        cbar.ax.tick_params(labelsize=font_size, length=6)
        cbar.set_label(label=r'$\sigma^2_{\phi}/\sigma^2_0$', size=font_size)
        
        chosen_Mx = sigma.shape[1]
        chosen_My = sigma.shape[0]

        plt.xlabel(r'$M_x$', fontsize=font_size)
        plt.ylabel(r'$M_y$', fontsize=font_size)

        plt.xticks([0, chosen_Mx/4, chosen_Mx/2, 3*chosen_Mx/4, chosen_Mx], 
                   [r'0', str(int(chosen_Mx/4)), str(int(chosen_Mx/2)), str(int(3*chosen_Mx/4)), str(int(chosen_Mx))], fontsize=font_size)
        plt.yticks([0, chosen_My/4, chosen_My/2, 3*chosen_My/4, chosen_My], 
                   [r'0', str(int(chosen_My/4)), str(int(chosen_My/2)), str(int(3*chosen_My/4)), str(int(chosen_My))], fontsize=font_size)

        plt.tick_params(axis='x', direction='in', length=6, top=True)
        plt.tick_params(axis='y', direction='in', length=6, right=True)
        plt.tick_params(which='minor', axis='y', direction='in', right=True)


        in_object_g = h_in.real_time(params=self.param_dict)
        folder_name_plot_g, file_name_plot_green = in_object_g.plot_polaron_size_folder_structure_real_time_prop(path_main, t_index)

        plt.savefig(folder_name_plot_g+file_name_plot_green+'.png', dpi=400, bbox_inches='tight')
        plt.close()

    def plot_polaron_size_imag_time(self, sigma, V_0, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        fig = plt.figure()
        plt.title(r'$V_0 =$'+str(V_0), fontsize=font_size)

        pc = plt.pcolormesh(sigma, vmin=0, vmax=1)
        cbar = fig.colorbar(pc)
        cbar.ax.tick_params(labelsize=font_size, length=6)
        cbar.set_label(label=r'$\sigma^2_{\phi}/\sigma^2_0$', size=font_size)
        
        chosen_Mx = sigma.shape[1]
        chosen_My = sigma.shape[0]

        plt.xlabel(r'$M_x$', fontsize=font_size)
        plt.ylabel(r'$M_y$', fontsize=font_size)

        plt.xticks([0, chosen_Mx/4, chosen_Mx/2, 3*chosen_Mx/4, chosen_Mx], 
                   [r'0', str(int(chosen_Mx/4)), str(int(chosen_Mx/2)), str(int(3*chosen_Mx/4)), str(int(chosen_Mx))], fontsize=font_size)
        plt.yticks([0, chosen_My/4, chosen_My/2, 3*chosen_My/4, chosen_My], 
                   [r'0', str(int(chosen_My/4)), str(int(chosen_My/2)), str(int(3*chosen_My/4)), str(int(chosen_My))], fontsize=font_size)

        plt.tick_params(axis='x', direction='in', length=6, top=True)
        plt.tick_params(axis='y', direction='in', length=6, right=True)
        plt.tick_params(which='minor', axis='y', direction='in', right=True)

        in_object_g = h_in.imag_time(params=self.param_dict)
        folder_name_plot, file_name_plot = in_object_g.polaron_size_results_folder_structure_imag_time_prop(path_main)

        plt.savefig(folder_name_plot+file_name_plot+str(V_0)+'.png', dpi=400, bbox_inches='tight')
        plt.close()

    # plot polaron size for a specific potential - mesh on the grid
    def plot_heff_matrix(self, matrix, V_0, params_wfs, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        fig = plt.figure()
        plt.title(r'$V_0 =$'+str(V_0), fontsize=font_size)

        pc = plt.pcolormesh(matrix[::-1,:])
        cbar = fig.colorbar(pc)
        cbar.ax.tick_params(labelsize=font_size, length=6)
        cbar.set_label(label=r'$\hat{H}_{eff}$', size=font_size)
        
        chosen_Mx = matrix.shape[1]
        chosen_My = matrix.shape[0]

        plt.xlabel(r'states', fontsize=font_size)
        plt.ylabel(r'states', fontsize=font_size)

        plt.xticks([0.5, 1.5, 2.5, 3.5], 
                   [r'FO', r'FD v.', r'FD H.', r'SP'], fontsize=font_size)
        plt.yticks([0.5, 1.5, 2.5, 3.5], 
                   [r'SP', r'FD h.', r'FD v.', r'FO'], fontsize=font_size)

        plt.tick_params(axis='x', direction='in', length=6, labeltop=True, labelbottom=False, top=True)
        plt.tick_params(axis='y', direction='in', length=6, right=True)
        plt.tick_params(which='minor', axis='y', direction='in', right=True)

        in_object_g = h_in.coupl_states(params_calc=self.param_dict, params_wfs=params_wfs)
        folder_name_plot, file_name_plot_heff, file_name_plot_s_overlap = in_object_g.matrices_results_coupling_of_states(path_main)

        plt.savefig(folder_name_plot+file_name_plot_heff+str(V_0)+'.png', dpi=400, bbox_inches='tight')
        plt.close()

    # plot polaron size for a specific potential - mesh on the grid
    def plot_s_overlap_matrix(self, matrix, V_0, params_wfs, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        fig = plt.figure()
        plt.title(r'$V_0 =$'+str(V_0), fontsize=font_size)

        pc = plt.pcolormesh(matrix[::-1,:])
        cbar = fig.colorbar(pc)
        cbar.ax.tick_params(labelsize=font_size, length=6)
        cbar.set_label(label=r'$\hat{S}$', size=font_size)
        
        chosen_Mx = matrix.shape[1]
        chosen_My = matrix.shape[0]

        plt.xlabel(r'states', fontsize=font_size)
        plt.ylabel(r'states', fontsize=font_size)

        plt.xticks([0.5, 1.5, 2.5, 3.5], 
                   [r'FO', r'FD v.', r'FD H.', r'SP'], fontsize=font_size)
        plt.yticks([0.5, 1.5, 2.5, 3.5], 
                   [r'SP', r'FD h.', r'FD v.', r'FO'], fontsize=font_size)

        plt.tick_params(axis='x', direction='in', length=6, labeltop=True, labelbottom=False, top=True)
        plt.tick_params(axis='y', direction='in', length=6, right=True)
        plt.tick_params(which='minor', axis='y', direction='in', right=True)

        in_object_g = h_in.coupl_states(params_calc=self.param_dict, params_wfs=params_wfs)
        folder_name_plot, file_name_plot_heff, file_name_plot_s_overlap = in_object_g.matrices_results_coupling_of_states(path_main)

        plt.savefig(folder_name_plot+file_name_plot_s_overlap+str(V_0)+'.png', dpi=400, bbox_inches='tight')
        plt.close()

    def plot_configuration(self, psi_rotors, V_0_pool, V_index, scan_dir, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        M = int(self.Mx*self.My)

        psi_rotors = psi_rotors.reshape((len(V_0_pool), self.My,self.Mx,self.n))

        fig, axs = plt.subplots(self.My,self.Mx, subplot_kw=dict(polar=True))
        plt.suptitle(r"$V_0$ ="+str(V_0_pool[V_index]), fontsize=font_size) #+r" from scan: $V_0$ = "+str(V_0_pool[0])+r" - "+str(V_0_pool[len(V_0_pool)-1]))

        for i in range(self.My):
            for j in range(self.Mx):
                rotor_density = (np.conjugate(psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx])\
                    *psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx]).T
                #rotor_density = np.sign(psi_rotors[V_index,(i+int(My/2))%My,(j+int(Mx/2))%Mx].imag)*np.arccos(psi_rotors[V_index,(i+int(My/2))%My,(j+int(Mx/2))%Mx].real/np.abs(psi_rotors[V_index,(i+int(My/2))%My,(j+int(Mx/2))%Mx]))

                axs[i, j].plot(self.x, rotor_density.real)

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_theta_zero_location('E')

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(self.Vmin)+'_Vmax_'+str(self.Vmax)+'_complete/configurations/'
        file_name   = 'psi_rotors_2d_configuration_V_0_'+str(V_0_pool[V_index])+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_scan_direction_'+scan_dir
        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.svg') #, dpi=400)

        plt.show()

    def plot_phase(self, psi_rotors, V_0_pool, V_index, scan_dir, path_main):
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #plt.rc('text.latex', preambler=r'\usepackage{textgreek}')
        font_size = 18 

        M = int(self.Mx*self.My)

        psi_rotors = psi_rotors.reshape((len(V_0_pool), self.My,self.Mx,self.n))

        fig, axs = plt.subplots(self.My,self.Mx, subplot_kw=dict(polar=True))
        plt.suptitle(r"$V_0$ ="+str(V_0_pool[V_index])) #+r" from scan: $V_0$ = "+str(V_0_pool[0])+r" - "+str(V_0_pool[len(V_0_pool)-1]))

        for i in range(self.My):
            for j in range(self.Mx):
                rotor_phase = np.sign(psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx].imag)\
                    *np.arccos(psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx].real\
                        /np.abs(psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx]))

                axs[i, j].plot(self.x, rotor_phase)

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_theta_zero_location('E')

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(self.Vmin)+'_Vmax_'+str(self.Vmax)+'_complete/configurations/'
        file_name   = 'psi_rotors_2d_phase_V_0_'+str(V_0_pool[V_index])+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_scan_direction_'+scan_dir
        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)

        plt.show()

class densities:
    def __init__(self, Mx, My, B, V_0, tx, ty, qx, qy, n, dt, tol):
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.V_0 = V_0
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.x   = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid
        self.dt  = dt
        self.tol = tol

    def plot_densities(self, psi_rotors, V_0_pool, scan_dir):
        M = int(self.Mx*self.My)

        potential_points = len(V_0_pool)

        psi_rotors = psi_rotors.reshape((potential_points,self.My,self.Mx,self.n))

        fig, axs = plt.subplots(self.My,self.Mx, sharex=True, sharey=True)
        plt.suptitle(r"scan: $V_0$ = "+str(V_0_pool[0])+r" - "+str(V_0_pool[len(V_0_pool)-1]))

        for i in range(self.My):
            for j in range(self.Mx):
                rotor_density = (np.conjugate(psi_rotors[:,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx])\
                    *psi_rotors[:,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx]).T

                axs[i,j].pcolormesh(rotor_density)

                axs[i,j].set_yticks([0,self.n/2,self.n],['0','1','2'])
                axs[i,j].set_xticks([0,potential_points/2,potential_points]) # [0,potential_points/4,potential_points/2,3*potential_points/4,potential_points] #,[0,str(potential_points/4),str(potential_points/2),str(3*potential_points/4),str(potential_points)])
                if j == 0:
                    axs[i,j].set_ylabel(r'$\phi/\pi$')
                if i == int(M**0.5) - 1:
                    axs[i,j].set_xlabel(r'$V_0/t$')

        Vmin = V_0_pool[0]
        Vmax = V_0_pool[len(V_0_pool)-1]
        
        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'/densities/'
        file_name   = 'psi_rotors_2d_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_scan_direction_'+scan_dir
        plt.savefig(in_object.get_file_name(folder_name, file_name)+'.png', dpi=400)
        plt.show()

    def plot_configuration(self, psi_rotors, V_0_pool, V_index, scan_dir):
        M = int(self.Mx*self.My)

        potential_points = len(V_0_pool)

        psi_rotors = psi_rotors.reshape((potential_points,self.My,self.Mx,self.n))

        fig, axs = plt.subplots(self.My,self.Mx, subplot_kw=dict(polar=True))
        plt.suptitle(r"$V_0$ ="+str(V_0_pool[V_index])+r" from scan: $V_0$ = "+str(V_0_pool[0])+r" - "+str(V_0_pool[len(V_0_pool)-1]))

        for i in range(self.My):
            for j in range(self.Mx):
                rotor_density = (np.conjugate(psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx])\
                    *psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx]).T
                #rotor_density = np.sign(psi_rotors[V_index,(i+int(My/2))%My,(j+int(Mx/2))%Mx].imag)*np.arccos(psi_rotors[V_index,(i+int(My/2))%My,(j+int(Mx/2))%Mx].real/np.abs(psi_rotors[V_index,(i+int(My/2))%My,(j+int(Mx/2))%Mx]))

                axs[i, j].plot(self.x, rotor_density.real)

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_theta_zero_location('E')
        
        Vmin = V_0_pool[0]
        Vmax = V_0_pool[len(V_0_pool)-1]
        
        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'/configurations/'
        file_name   = 'psi_rotors_2d_configuration_V_0_'+str(V_0_pool[V_index])+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_scan_direction_'+scan_dir
        
        plt.savefig(in_object.get_file_name(folder_name, file_name)+'.svg') #, dpi=400)

        plt.show()
    
    def plot_phases(self, psi_rotors, V_0_pool, V_index, scan_dir):
        M = int(self.Mx*self.My)

        potential_points = len(V_0_pool)

        psi_rotors = psi_rotors.reshape((potential_points,self.My,self.Mx,self.n))

        fig, axs = plt.subplots(self.My,self.Mx, subplot_kw=dict(polar=True))
        plt.suptitle(r"$V_0$ ="+str(V_0_pool[V_index])+r" from scan: $V_0$ = "+str(V_0_pool[0])+r" - "+str(V_0_pool[len(V_0_pool)-1]))

        for i in range(self.My):
            for j in range(self.Mx):
                rotor_phase = np.sign(psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx].imag)\
                    *np.arccos(psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx].real/np.abs(psi_rotors[V_index,(i+int(self.My/2))%self.My,(j+int(self.Mx/2))%self.Mx]))

                axs[i, j].plot(self.x, rotor_phase)

                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_theta_zero_location('E')
        
        Vmin = V_0_pool[0]
        Vmax = V_0_pool[len(V_0_pool)-1]
        
        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'/configurations/'
        file_name   = 'psi_rotors_2d_pase_V_0_'+str(V_0_pool[V_index])+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_scan_direction_'+scan_dir
        plt.savefig(in_object.get_file_name(folder_name, file_name)+'.png', dpi=400)

        plt.show()

class energies:
    def __init__(self, Mx, My, B, tx, ty, qx, qy, n, x, dt, tol):
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

    # new function for plotting the crossing - is more general, in particular storing and calculation of energies is outsourced
    def plot_e_simple_crossing(self, V1, E1, V2, E2, Vmin, Vmax, path_main):
        M = int(self.Mx*self.My)

        scale = np.max(np.array([self.tx,self.ty])) # give all quantities relative to the maximum of the maximum tunneling

        ax = plt.gca()
        plt.suptitle(r"$B$ = "+str(self.B)+r", $t_x/t_y$ = "+str(self.tx)+'/'+str(self.ty)) # plot title

        # plot energies for comparison
        plt.scatter(V1/scale, E1/scale, marker='x', label='forward scan')
        plt.scatter(V2/scale, E2/scale, marker='x', label='backward scan')

        # formatter of the axis
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    
        ax.yaxis.set_major_formatter(OOMFormatter(0, "%1.1f"))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
        plt.xlabel(r'$V_0/$max($t_x,t_y$)')
        plt.ylabel(r'$E/$max($t_x,t_y$)')
        plt.legend()

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+\
            '_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'_complete/energies/'
        file_name   = 'energ_2d_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)
        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)

        plt.show() 

    # old plot function - don't delete to ensure backward compatibility
    def plot_energies_simple_crossing(self, psi_rotors_f, psi_rotors_b, Vmin, Vmax, Vback, potential_points, pot_points_back, path_main):
        M = int(self.Mx*self.My)

        V_0_pool_f = np.linspace(Vmin, Vmax, potential_points) # for forward scanning
        V_0_pool_b = np.linspace(Vmax, Vback, pot_points_back)[::-1] # for backward scanning

        energy_object = energy.energy(Mx=self.Mx, My=self.My, B=self.B, V_0=0, tx=self.tx, ty=self.ty,
                                      qx=self.qx, qy=self.qy, n=self.n, x=self.x, dt=self.dt, tol=self.dt)

        E_col_f = np.zeros(len(V_0_pool_f), dtype=complex)
        E_col_b = np.zeros(len(V_0_pool_b), dtype=complex)

        # compute the energies
        i = 0
        for V_0_f in V_0_pool_f:
            energy_object.V_0 = V_0_f
            E_col_f[i], E_T, E_B, E_V = energy_object.calc_energy(psi_rotors_f[i].reshape((self.Mx,self.My,self.n)))
            i += 1

        i = 0
        for V_0_b in V_0_pool_b:
            energy_object.V_0 = V_0_b
            E_col_b[i], E_T, E_B, E_V = energy_object.calc_energy(psi_rotors_b[i].reshape((self.Mx,self.My,self.n)))
            i += 1

        #print('E_b - E_f =', (E_col_b[::-1] - E_col_f))
        #print('E_b - E_m =', (E_col_b[::-1] - scipy.special.mathieu_a(0, 2*np.linspace(0, 400, 41))))

        scale = np.max(np.array([self.tx,self.ty])) # give all quantities relative to the maximum of the maximum tunneling

        ax = plt.gca()
        plt.suptitle(r"$B$ = "+str(self.B)+r", $t_x/t_y$ = "+str(self.tx)+'/'+str(self.ty))

        # plot energies for comparison
        plt.scatter(V_0_pool_b/scale, E_col_b.real/scale, marker='x', label='backward scanning')
        plt.scatter(V_0_pool_f/scale, E_col_f.real/scale, marker='x', label='forward scanning')

        plot_mathieu = input('Plot Mathieu Energies (y/n)? ')
        if plot_mathieu == 'y':
            x_mathieu = np.linspace(Vback, Vmax, 1000)
            y_mathieu = scipy.special.mathieu_a(0, 2*x_mathieu)
            plt.plot(x_mathieu/scale, y_mathieu/scale, linestyle='dotted', linewidth=1, color='red', label='mathieu energies')

        # formatter of the axis
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    
        ax.yaxis.set_major_formatter(OOMFormatter(0, "%1.1f"))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
        plt.xlabel(r'$V_0/$max($t_x,t_y$)')
        plt.ylabel(r'$E/$max($t_x,t_y$)')
        plt.legend()

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'_complete/energies/'
        file_name   = 'energ_2d_configuration_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)
        #plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)

        #np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_forward.out', (V_0_pool_f, E_col_f))
        #np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_backward.out', (V_0_pool_b, E_col_b))

        plt.show()

    def plot_resolved_coupling(self, n_states, e1, e2, e_vals, Vmin, Vmax, potential_points, path_main):
        M = int(self.Mx*self.My)

        scale = np.max(np.array([self.tx,self.ty])) # give all quantities relative to the maximum of the maximum tunneling
        V_0_pool = np.linspace(Vmin, Vmax, potential_points)

        for i in range(n_states):
            plt.plot(V_0_pool/scale, e_vals.T[i].real/scale, marker='x')

        plt.plot(V_0_pool/scale, e1/scale, label=r'state 1')
        plt.plot(V_0_pool/scale, e2/scale, label=r'state 2')

        plt.xlabel(r'$V_0/$max($t_x,t_y$)')
        plt.ylabel(r'$E/$max($t_x,t_y$)')
        plt.legend() 
        
        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'_complete/coupling_of_states/'
        file_name   = 'energ_coupling_2d_configuration_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)
        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)

        #for i in range(n_states):
        #    np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_state_'+str(i)+'.out', (V_0_pool, e_vals.T[i]))
        
        #np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_normal_state_1.out', (V_0_pool, e1))
        #np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_normal_state_2.out', (V_0_pool, e2))
        
        plt.show()

    def plot_resolved_coupling_state_contributions(self, n_states, e_kets, Vmin, Vmax, potential_points, path_main):
        M = int(self.Mx*self.My)

        scale = np.max(np.array([self.tx,self.ty])) # give all quantities relative to the maximum of the maximum tunneling
        V_0_pool = np.linspace(Vmin, Vmax, potential_points)

        a = np.zeros((potential_points,n_states))
        a2 = np.zeros(n_states)
        for i in range(n_states):
            for j in range(n_states):
                a[:,j] = (np.conjugate(e_kets.T[i][j])*e_kets.T[i][j]).real
         #       print((np.conjugate(e_kets.T[i][j])*e_kets.T[i][j]).real)
                #plt.plot(V_0_pool/scale, (np.conjugate(e_kets.T[i][j])*e_kets.T[i][j]).real, marker='x', label=r'state '+str(i+1))
            plt.plot(V_0_pool/scale, a.T[i], marker='x', label=r'state '+str(i+1))

        #print(a.T.shape)
        plt.plot(V_0_pool/scale, (np.conjugate(e_kets.T[0][0])*e_kets.T[0][0]+np.conjugate(e_kets.T[1][1])*e_kets.T[1][1]).real, marker='x')
        #plt.plot(V_0_pool/scale, np.sum(a, axis=1), marker='x')


        plt.xlabel(r'$V_0/$max($t_x,t_y$)')
        plt.ylabel(r'$E/$max($t_x,t_y$)')
        plt.legend() 

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_Vmin_'+str(Vmin)+'_Vmax_'+str(Vmax)+'/energies/'
        file_name   = 'energ_coupling_contribution_states_2d_configuration_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)
        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)

        #for i in range(n_states):
        #    np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_state_'+str(i)+'.out', (V_0_pool, ekts.T[i]))

        plt.show()

    def plot_phase_transition(self, V, psi, path_main):
        M = int(self.Mx*self.My)

        energy_object = energy.energy(Mx=self.Mx, My=self.My, B=self.B, V_0=0, tx=self.tx, ty=self.ty,
                                      qx=self.qx, qy=self.qy, n=self.n, x=self.x, dt=self.dt, tol=self.tol)
        
        phase_number = len(V)
        phase_label  = np.array([r'phase I', r'phase II', r'phase III'])

        # compute the energies
        if phase_number == 3:
            E_col = np.zeros((phase_number,np.max(np.array([len(V[0]),len(V[1]),len(V[2])]))))
        elif phase_number == 2:
            E_col = np.zeros((phase_number,np.max(np.array([len(V[0]),len(V[1])]))))

        for i in range(phase_number):
            V_i = V[i]
            for j in range(len(V_i)):
                energy_object.V_0 = V_i[j]
                E_col[i][j] = energy_object.calc_energy(psi[i][j])[0].real

        ax = plt.gca()
        plt.suptitle(r"$B$ = "+str(self.B)+r", $t_x/t_y$ = "+str(self.tx)+'/'+str(self.ty))

        # plot energies of the phases
        scale = np.max(np.array([self.tx,self.ty]))
        for i in range(len(E_col)):
            print(E_col[i])
            plt.scatter(V[i]/scale, E_col[i][E_col[i] != 0]/scale, marker='x', label=phase_label[i])

        print(' ')
        plot_mathieu = input('Plot Mathieu Energies (y/n)? ')
        if plot_mathieu == 'y':
            Vback = V[phase_number-2][int(3*len(V[phase_number-2])/4)]
            Vmax  = V[phase_number-1][len(V[phase_number-1])-1]
            x_mathieu = np.linspace(Vback, Vmax, 1000)
            y_mathieu = scipy.special.mathieu_a(0, 2*x_mathieu)
            plt.plot(x_mathieu/scale, y_mathieu/scale, linestyle='dotted', linewidth=1, color='red', label='mathieu energies')

        # formatter of the axis
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    
        ax.yaxis.set_major_formatter(OOMFormatter(0, "%1.1f"))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
        ax_t = ax.secondary_xaxis('top')

        # plot the points of the phase transition on top, as a secondary axis
        tick_list = []
        tick_label_list = []
        for i in range(len(E_col)-1):
            x_transition_pos = (V[i][len(V[i])-1]+V[i+1][0])/(2*scale)
            plt.axvline(x=x_transition_pos, linestyle='dashed')
            tick_list.append(x_transition_pos)
            tick_label_list.append(r'$\approx$ '+str(x_transition_pos))
        ax_t.set_xticks(tick_list, tick_label_list) 

        #plt.ylim(-1300/scale,-100/scale)

        plt.xlabel(r'$V_0/$max($t_x,t_y$)')
        plt.ylabel(r'$E/$max($t_x,t_y$)')
        plt.legend(fontsize=7)

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(V[0][0])+'_Vmax_'+str(V[len(V)-1][len(V[len(V)-1])-1])+'_complete/energies/'
        file_name   = 'energ_2d_phase_transition_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_qx_'+str(self.qx)+'_qy_'+str(self.qy)\
            +'_tol_'+str(self.tol)+'_dt_'+str(self.dt)

        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)

        for i in range(len(E_col)):
            print(V[i])
            np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'_phase'+str(i+1)+'.out', (V[i], E_col[i][E_col[i] != 0]))

        plt.show()

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

    def plot_effective_masses(self, V, mx, my, yscale, choice_scan_dir, path_main):
        M = int(self.Mx*self.My)

        scale = np.max(np.array([self.tx,self.ty]))

        plt.suptitle(r"$B$ = "+str(self.B)+r", $t_x/t_y$ = "+str(self.tx)+'/'+str(self.ty))

        plt.plot(V/scale, mx, marker='x', color='black', label=r'$m^*_{x,p}/m_{0,x}^*$')
        plt.plot(V/scale, my, marker='x', color='red', label=r'$m^*_{y,p}/m_{0,y}^*$')

        plt.yscale(yscale)

        plt.xlabel(r'$V_0/$max($t_x,t_y$)')
        plt.ylabel(r'$m_i^*/m_0^*$')
        plt.legend()
        
        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line
        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(np.min(V))+'_Vmax_'+str(np.max(V))+'_complete/effective_mass/'
        file_name   = 'eff_mass_2d_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)+'_'+choice_scan_dir

        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)
        plt.show()
    
    def plot_eff_mass_energies(self, V, E_col, E_col_qx, E_col_qy, choice_scan_dir, path_main):
        M = int(self.Mx*self.My)

        scale = np.max(np.array([self.tx,self.ty]))

        ax = plt.gca()

        plt.suptitle(r"$B$ = "+str(self.B)+r", $t_x/t_y$ = "+str(self.tx)+'/'+str(self.ty))
    
        plt.scatter(V/scale, E_col/scale, marker='x', label=r'$q_x = q_y = 0$')
        plt.scatter(V/scale, E_col_qx/scale, marker='x', label=r'$q_x = \pm 1, q_y = 0$')
        plt.scatter(V/scale, E_col_qy/scale, marker='x', label=r'$q_x = 0, q_y = \pm 1$')

        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    
        ax.yaxis.set_major_formatter(OOMFormatter(0, "%1.1f"))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
        plt.xlabel(r'$V_0/$max($t_x,t_y$)')
        plt.ylabel(r'$E/$max($t_x,t_y$)')
        plt.legend(fontsize=7)

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(np.min(V))+'_Vmax_'+str(np.max(V))+'_complete/effective_mass/'
        file_name   = 'energ_2d_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)+'_different_q_tol_'+str(self.tol)+'_dt_'+str(self.dt)+\
            '_'+choice_scan_dir
        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)

        plt.show()

class polaron_size:
    def __init__(self, Mx, My, B, V_0, tx, ty, qx, qy, n, dt, tol):
        self.Mx  = Mx
        self.My  = My
        self.B   = B
        self.V_0 = V_0 # again should be an array
        self.tx  = tx
        self.ty  = ty
        self.qx  = qx
        self.qy  = qy
        self.n   = n
        self.x   = (2*np.pi/n)*np.arange(n) # make phi (=angle) grid
        self.dt  = dt
        self.tol = tol

    # plot polaron size for a specific potential - mesh on the grid
    def plot_polaron_size(self, sigma, V_index, path_main, hide_show):
        M = int(self.Mx*self.My)

        fig = plt.figure()
        plt.suptitle(r"$V_0 =$ "+str(self.V_0[V_index])+r", $B =$ "+str(self.B))
        
        #sigma = sigma[40:60, 8:20]

        pc = plt.pcolormesh(sigma)
        fig.colorbar(pc)

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(self.V_0[0])+'_Vmax_'+str(self.V_0[len(self.V_0)-1])+'_complete/polaron_size/qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_mesh/'
        file_name   = 'polaron_size_2d_V_0_'+str(self.V_0[V_index])+'_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)

        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', dpi=400)

        np.savetxt(in_object.get_file_name(path_main, folder_name, file_name)+'.out', (sigma))

        if hide_show == False:
            plt.show()
        else:
            plt.close()

    def linear(self, ray_indices, start_indices, x):
        if ray_indices[0] == 0:
            k = np.inf
        else:
            k = ray_indices[1]/ray_indices[0]
        d = start_indices[1] - k*start_indices[0]
        return k*x + d

    # plot polaron size along a specific line in the configuration
    def plot_polaron_size_along_line(self, sigma_line_tot, sigma_lines, V_index, ray_indices, start_indices, path_main):
        M = int(self.Mx*self.My)

        fig, axs = plt.subplots(1,3)

        x = np.linspace(start_indices[0], self.Mx, 1000)
        y = self.linear(ray_indices, start_indices, x)

        # plot grid
        axs[0].plot(x, y, linestyle='dashed')
        axs[0].scatter(self.Mx/2-0.5, self.My/2-0.5, color='red', marker='x')
        for i in range(self.My):
            axs[0].scatter(np.arange(0, self.Mx), i*np.ones(self.Mx), marker='o', color='black')

        axs[0].set( aspect='equal')
        axs[0].title.set_text('Lattice')
        axs[0].set_xlim(-1, self.Mx)
        axs[0].set_ylim(-1, self.My)
        axs[0].set_xticks([0, (self.Mx-1)/2, self.Mx-1], ['-'+str(int(self.Mx/2)), '0', '+'+str(int(self.Mx/2))])
        axs[0].set_yticks([0, (self.My-1)/2, self.My-1], ['-'+str(int(self.My/2)), '0', '+'+str(int(self.My/2))])
        axs[0].set_xticks([], [])
        axs[0].set_yticks([], [])
        
        axs[0].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$y$')

        # plot polaron size for selected potential points
        for i in range(len(V_index)):
            axs[1].plot(sigma_lines[i], marker='x', label=r'$V_0 =$'+str(self.V_0[V_index[i]]))

        axs[1].set_xticks([0,len(sigma_lines[0])-1], ['line start', 'line end'])
        axs[1].set_ylabel(r'$\sigma$')
        axs[1].title.set_text(r'$\sigma$ for selected $V_0$')

        # colormesh
        pc = axs[2].pcolormesh(sigma_line_tot)
        fig.colorbar(pc, label=r'$\sigma$', ticks=[0,0.2,0.4,0.6,0.8,1], boundaries=np.linspace(0,1,100)) # ,  aspect=5.4 np.min(sigma_line_tot)
        
        axs[2].title.set_text(r'$\sigma$ for all $V_0$')
        axs[2].set_xticks([0,len(sigma_lines[0])], ['line start', 'line end'])
        axs[2].set_yticks([np.min(self.V_0),len(sigma_line_tot)/2,len(sigma_line_tot)], [str(np.min(self.V_0)),str(self.V_0[len(self.V_0)-1]/2),str(self.V_0[len(self.V_0)-1])])
        axs[2].set_ylabel(r'$V_0$')

        in_object = h_in.params(on_cluster=False) # object for handling inputs from command line

        folder_name = 'image_results/psi_rotors_2d_python_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_Vmin_'+str(self.V_0[0])+'_Vmax_'+str(self.V_0[len(self.V_0)-1])+'_complete/polaron_size/qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_along_line/'
        file_name   = 'polaron_size_2d_fig_all_comparison_V_0_'+str(self.V_0[V_index])+'_ray_indices_'+str(ray_indices)+'_start_indices_'+str(start_indices)\
            +'_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)

        plt.tight_layout()
        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', bbox_inches = 'tight', dpi=400)
        plt.show()


        # individual plots        
        # plot grid
        fig = plt.gca()

        plt.plot(x, y, linestyle='dashed')
        plt.scatter(self.Mx/2-0.5, self.My/2-0.5, color='red', marker='x')
        for i in range(self.My):
            plt.scatter(np.arange(0, self.Mx), i*np.ones(self.Mx), marker='o', color='black')

        plt.xlim(-1, self.Mx)
        plt.ylim(-1, self.My)
        plt.xticks([0, (self.Mx-1)/2, self.Mx-1], ['-'+str(int(self.Mx/2)), '0', '+'+str(int(self.Mx/2))])
        plt.yticks([0, (self.My-1)/2, self.My-1], ['-'+str(int(self.My/2)), '0', '+'+str(int(self.My/2))])
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')

        file_name   = 'polaron_size_2d_fig_1_lattice_comparison_V_0_'+str(self.V_0[V_index])+'_ray_indices_'+str(ray_indices)+'_start_indices_'+str(start_indices)\
            +'_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)

        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', bbox_inches='tight', dpi=400) 
        plt.close()

        
        # plot polaron size for selected potential points
        fig = plt.gca()
        for i in range(len(V_index)):
            plt.plot(sigma_lines[i], marker='x', label=r'$V_0 =$'+str(self.V_0[V_index[i]]))

        plt.xticks([0,len(sigma_lines[0])-1], ['line start', 'line end'])
        plt.xlabel(r'rotors along line')
        plt.ylabel(r'$\sigma$')

        plt.legend(bbox_to_anchor=(1.25, 1.0))

        file_name   = 'polaron_size_2d_fig_2_pot_s_comparison_V_0_'+str(self.V_0[V_index])+'_ray_indices_'+str(ray_indices)+'_start_indices_'+str(start_indices)\
            +'_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)

        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', bbox_inches='tight', dpi=400) 
        plt.close()


        # colormesh
        axs = plt.gca()

        pc = plt.pcolormesh(sigma_line_tot)
        plt.colorbar(pc, label=r'$\sigma$', ticks=[0,0.2,0.4,0.6,0.8,1], boundaries=np.linspace(0,1,100)) # np.min(sigma_line_tot)

        plt.xticks([0,len(sigma_lines[0])], ['line start', 'line end'])
        plt.yticks([np.min(self.V_0),len(sigma_line_tot)/2,len(sigma_line_tot)], [str(np.min(self.V_0)),str(self.V_0[len(self.V_0)-1]/2),str(self.V_0[len(self.V_0)-1])])
        plt.xlabel(r'rotors along line')
        plt.ylabel(r'$V_0$')

        file_name   = 'polaron_size_2d_fig_3_pot_a_comparison_V_0_'+str(self.V_0[V_index])+'_ray_indices_'+str(ray_indices)+'_start_indices_'+str(start_indices)\
            +'_M_'+str(M)+'_B_'+str(self.B)+'_tx_'+str(self.tx)+'_ty_'+str(self.ty)\
            +'_qx_'+str(self.qx)+'_qy_'+str(self.qy)+'_tol_'+str(self.tol)+'_dt_'+str(self.dt)

        plt.savefig(in_object.get_file_name(path_main, folder_name, file_name)+'.png', bbox_inches='tight', dpi=400)
        plt.close()