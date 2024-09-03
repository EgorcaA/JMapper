from triqs.gf import Gf, MeshImFreq
from triqs.gf import iOmega_n, inverse

import numpy as np
from matplotlib.ticker import (MultipleLocator,  AutoMinorLocator)
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from  tqdm import tqdm



class Exchanges2():
    '''
    class for calculating exchange integrals in Heisenberg model for one atom in cell
    '''

    def __init__(self, hk_mn_up, hk_mn_dn, DeltaBlock, S, eF=0.0, beta=10, n_iw=200):
        '''
        hks     - wannier hamiltonian, 
        eF      - Fermi level
        DeltaBlock
        S       - spin value
        S  ~= 1.5 for CrTe2
        
        '''
        self.nwa = hk_mn_dn.shape[1]
        self.nkpt = hk_mn_dn.shape[0]
        
        self.nkpt_dir = int(self.nkpt**0.5)
        self.kpoints_adj_serial = np.mgrid[0:1:1.0/self.nkpt_dir, 0:1:1.0/self.nkpt_dir].reshape(2,-1).T
        self.dSarea = 1.0/(self.nkpt)

        self.DeltaBlock = DeltaBlock
        self.beta = beta
        self.S = S

        iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=n_iw)
        Giws_up = [Gf(mesh=iw_mesh, target_shape=[self.nwa, self.nwa]) for kpt in range(self.nkpt)]
        Giws_dn = [Gf(mesh=iw_mesh, target_shape=[self.nwa, self.nwa]) for kpt in range(self.nkpt)]

        for kpt in tqdm(range(len(hk_mn_up))):
            Giws_up[kpt] << inverse(iOmega_n  + eF*np.eye(self.nwa) - hk_mn_up[kpt])
            Giws_up[kpt] = Giws_up[kpt].data
        Giws_up = np.array(Giws_up)

        for kpt in tqdm(range(len(hk_mn_dn))):
            Giws_dn[kpt] << inverse(iOmega_n  + eF*np.eye(self.nwa) - hk_mn_dn[kpt])
            Giws_dn[kpt] = Giws_dn[kpt].data
        Giws_dn = np.array(Giws_dn)


         
        '''
        G has form 2 x (2 x Te atoms p  + Cr atom d )
        [pp pp pp pp pd pd]
        [pp pp pp pp pd pd]
        [pp pp pp pp pd pd]
        [pp pp pp pp pd pd]
        [dp dp dp dp dd dd]
        [dp dp dp dp dd dd]
        '''
        dGiws_up = Giws_up[:, :, 12:, 12:]
        '''
        Now G has form
        [ dd dd]
        [ dd dd]
        '''
        Giws_upBlock = np.array([
            [dGiws_up[:, :, :5, :5], dGiws_up[:, :, :5, 5:]],
            [dGiws_up[:, :, 5:, :5], dGiws_up[:, :, 5:, 5:]]
        ])

        dGiws_dn = Giws_dn[:, :, 12:, 12:]
        Giws_dnBlock = np.array([
            [dGiws_dn[:, :, :5, :5], dGiws_dn[:, :, :5, 5:]],
            [dGiws_dn[:, :, 5:, :5], dGiws_dn[:, :, 5:, 5:]]
        ])

        self.Giws_up_flt = np.transpose(Giws_upBlock, (0, 1, 3, 4, 5, 2)).reshape(2, 2, -1, self.nkpt)
        self.Giws_dn_flt = np.transpose(Giws_dnBlock, (0, 1, 3, 4, 5, 2)).reshape(2, 2, -1, self.nkpt)
        self.Giws_upBlock = Giws_upBlock
        self.Giws_dnBlock = Giws_dnBlock

        self.Jself = self.calc_J(np.array([0, 0]), 0,0) # self interaction


    def calc_J(self,  vec, i, j, verb=True):
        '''
        vec     - direct vector to neighbour
        i, j    - index of atom in primitive cell
        '''
        dnwa = 5
        G_w_flt_up =  self.dSarea*np.sum( np.exp(-2*np.pi*1.j* np.dot(self.kpoints_adj_serial, vec) )*self.Giws_up_flt[i, j], axis=1 )  
        G_w_flt_dn =  self.dSarea*np.sum( np.exp(+2*np.pi*1.j* np.dot(self.kpoints_adj_serial, vec) )*self.Giws_dn_flt[j, i], axis=1 )  
        
        J = -0.25/self.beta*(
            np.sum(np.trace(self.DeltaBlock[i, i]@(G_w_flt_up.reshape(-1, dnwa , dnwa))@\
                            self.DeltaBlock[j ,j]@(G_w_flt_dn.reshape(-1, dnwa , dnwa)), axis1=1, axis2=2)
                    ))
        
        if verb:
            print(f'J = {1000*J*2/(self.S**2):.5f} meV')
            # adjusting for Heisenberg model -1/2 \sum_{i,j} J_{i j} S_i S_j
        return J*2/(self.S**2)


    def get_kqpt(self, kpt, q):
        '''
        shift flat indice (kpt) on target vector q 
        '''
        # flat indice to coordinate on square lattice
        kx = kpt//self.nkpt_dir
        ky = kpt - kx*self.nkpt_dir
        k = np.array([kx, ky])
        
        # shift on q
        kq = np.mod(k+q, self.nkpt_dir)
        kqpt = kq[0]*self.nkpt_dir + kq[1]
        return kqpt 

    def getJpath(self, kpath):
        self.Jqs00 = self._getJpath(0, 0, kpath)
        self.Jqs01 = self._getJpath(0, 1, kpath)
        return self.Jqs00, self.Jqs01

    def _getJpath(self, i, j, kpath):
        '''
        calculate J_{i j}(q) on kpath
        '''
        Jqs = []
        for q in tqdm(kpath):
            val = 0
            for kpt in range(self.nkpt):
                val = val - 0.25/self.beta*self.dSarea*np.sum(np.trace(self.DeltaBlock[i, i]@self.Giws_dnBlock[i, j, self.get_kqpt(kpt, q) ]\
                                                                @self.DeltaBlock[j, j]@self.Giws_upBlock[j, i, kpt], axis1=1, axis2=2), axis=0)
            Jqs.append(val)
            # print(f'J0 = {1000*val:.5f} meV')
        Jqs = np.array(Jqs)*2/(self.S**2)
        
        if i == j :
            print(f'J(Gamma) = {1000*np.real(Jqs[0]- self.Jself):.3f} meV')
        else:
            print(f'J(Gamma) = {1000*np.real(Jqs[0]):.3f} meV')

        return Jqs


    def plot(self, dists_ticks, label_ticks, kpath_draw, saveQ=False):
    
        fig, (dd) = plt.subplots()


        dd.plot(kpath_draw, 
                (self.Jqs00 - self.Jself)*1000,  color='red', linewidth=0.7,
                    alpha=1, marker=">",  markersize=3.0,  label=r"Re[$J_q'$]")

        dd.plot(kpath_draw,
                (np.real(self.Jqs01) )*1000,  color='red', linewidth=0.7,
                    alpha=1, marker="D",  markersize=3.0,  label=r"Re[$J_{01}(q)'$]")

        dd.plot(kpath_draw,
                (np.imag(self.Jqs01) )*1000,  color='blue', linewidth=0.7,
                    alpha=1, marker="o",  markersize=3.0,  label=r"Im[$J_{01}(q)'$]")


        dd.set_ylabel('J(q) (eV)')  # Add an x-label to the axes.
        dd.set_xlabel('q')  # Add a y-label to the axes.
        dd.set_title(rf'J(q) $\beta$={self.beta:.0f}')
        # dd.legend(prop={'size': 9}, frameon=False, loc='upper right', bbox_to_anchor=(0.9, 0.99))  # Add a legend.
        # locator = AutoMinorLocator()
        # dd.yaxis.set_minor_locator(MultipleLocator(0.05))

        dd.set_xticks(dists_ticks, label_ticks)
        dd.grid(axis='x')
        dd.xaxis.set_minor_locator(AutoMinorLocator())
        dd.tick_params(top=True, right=True, which='minor',length=2, width=0.2, direction="in")
        dd.tick_params(top=True, right=True, which='major',length=3.5, width=0.4, labelsize=8, direction="in")

        dd.set_xlim(dists_ticks[0], dists_ticks[-1])
        # dd.set_ylim(-0.3, 0.2)
        dd.text(-0.1, 1.0, 'a)', transform=dd.transAxes,
                fontsize=10, fontweight='normal', va='top', ha='right')


        # dd.yaxis.set_major_locator(MultipleLocator(0.1))

        plt.rcParams['axes.linewidth'] = 0.3
        
        fig.set_figwidth(6)     
        fig.set_figheight(5/1.6)
        fig.tight_layout()
        if saveQ:
            plt.savefig(f'./Jq_beta_{self.beta:.0f}.eps', 
                        format='eps', dpi=200, bbox_inches='tight')
        plt.show() 




def plotJ_Real(Js_neib00, Js_neib01, sorted_vert_J, J_path_plt):
    '''
    plots J(R)
    Js_neib00, Js_neib01   - J_00(R) and  J_01(R)  
    sorted_vert_J          - dict: dist <-> [coords], example {3.71: [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, 0], [1, 1]], ...}
    J_path_plt             - list of neighbours orders, example  [1, 1, 1, 1, 1, 1, ...]
    '''
    def print_stats(Js):
        tmp = 0
        total = 0
        plot_data = []
        print(f'\t #Neib   \t Sum J (meV)  \t J mean (meV)')
        print('-'*50)
        for order, order_dist in enumerate(sorted_vert_J.keys()):
            num_vert = len(sorted_vert_J[order_dist])
            J_params = Js[tmp:tmp + num_vert]*1000
            tmp = tmp + num_vert
            total += np.real(np.sum(J_params))
            print(f' \
                {num_vert}\t\
                {np.real(np.sum(J_params)):.2f}\t \
                {np.real(np.sum(J_params)/num_vert):.2f} \
                ')
            plot_data.append([order+1, np.real(np.sum(J_params)/num_vert)])
        plot_data = np.array(plot_data)

        print(f'Total {np.real(total):.2f} (meV) ' + 'FM' if np.real(total) > 0 else 'AFM')
        return plot_data

    print('\nJ00')
    plot_data00 = print_stats(Js_neib00)
    print('\nJ01')
    plot_data01 = print_stats(Js_neib01)


    fig, dd = plt.subplots()

    dd.plot(plot_data00[:, 0], plot_data00[:,1])
    dd.scatter(plot_data00[:, 0], plot_data00[:,1],color='black', 
                alpha=1, marker="D",  s=15.0, label=r'Re[$J_{00}(r)$]')

    dd.plot(plot_data01[:, 0], plot_data01[:,1])
    dd.scatter(plot_data01[:, 0], plot_data01[:,1],color='red', 
                alpha=1, marker="D",  s=15.0, label=r'Re[$J_{01}(r)$]')


    dd.set_ylabel('E (meV)')  # Add an x-label to the axes.
    dd.set_xlabel('Neighbour order')  # Add a y-label to the axes.
    # dd.legend(prop={'size': 9}, frameon=False)  # Add a legend.

    dd.xaxis.set_major_locator(MultipleLocator(1))
    dd.yaxis.set_minor_locator(MultipleLocator(0.5))
    dd.tick_params(top=False, right=False, which='minor',length=2, width=0.2, direction="in")
    dd.tick_params(top=False, right=False, which='major',length=3.5, width=0.4, labelsize=8, direction="in")
    dd.hlines(0, xmin=0, xmax=max(J_path_plt), colors='black', linewidth=0.4)
    dd.set_xlim(0.01, J_path_plt[-1])

    plt.rcParams['axes.linewidth'] = 0.5
    width = 5
    fig.set_figwidth(5)     #  ширина в дюймах (2,54)
    fig.set_figheight(5/1.6)    #  высота в дюймах (2,54)
    fig.tight_layout()
    # plt.savefig('./2pub/pics/J_r_beta_10.eps', 
    #             format='eps', dpi=200, bbox_inches='tight')

    plt.show()