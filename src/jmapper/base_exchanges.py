from abc import ABC, abstractmethod
from triqs.gf import Gf, MeshImFreq, iOmega_n, inverse
import numpy as np
import pickle 
from  tqdm import tqdm


class Exchanges(ABC):

    def __init__(self, Giws_upBlock, Giws_dnBlock, DeltaBlock, beta, S):
        '''
        hks     - wannier hamiltonian, 
        eF      - Fermi level
        Delta
        Giws_upBlock (num_sites, num_sites, nkpt, num_iw, nwa, nwa) 
            in fractional coords [0, 1] x ndim 
        S       - spin value
        kpoints_grid
        '''
        self.num_sites = Giws_upBlock.shape[0] 
        self.nwa = Giws_upBlock.shape[-1]
        self.nkpt = Giws_upBlock.shape[2] # 10*10*10 for example
        self.n_iw = Giws_upBlock.shape[3]
        self.beta = beta
        self.S = S
        self.dSarea = 1./self.nkpt



        self.Giws_upBlock = Giws_upBlock
        self.Giws_dnBlock = Giws_dnBlock
        self.DeltaBlock = DeltaBlock
        
        self.Giws_up_flt = np.transpose(Giws_upBlock, (0, 1, 3, 4, 5, 2)).reshape(self.num_sites, self.num_sites, -1, self.nkpt)
        self.Giws_dn_flt = np.transpose(Giws_dnBlock, (0, 1, 3, 4, 5, 2)).reshape(self.num_sites, self.num_sites, -1, self.nkpt)
        print(self.Giws_up_flt.shape)

    # @property
    # @abstractmethod
    # def kMesh_int(self):
    #     """Subclasses must define this variable"""
    #     pass

    # @property
    # @abstractmethod
    # def kMesh_frac(self):
    #     """Subclasses must define this variable"""
    #     pass

    def calc_J(self, vec, i, j, verb=True):
        
        dnwa = self.DeltaBlock[i, i].shape[0]

        G_w_flt_up =  self.dSarea*np.sum( np.exp(-2*np.pi*1.j* np.dot(self.kMesh_frac, vec ) )*self.Giws_up_flt[i, j], axis=1 )  
        G_w_flt_dn =  self.dSarea*np.sum( np.exp(+2*np.pi*1.j* np.dot(self.kMesh_frac, vec ) )*self.Giws_dn_flt[j, i], axis=1 )  
        # print("hellp")
        J = -1.0/self.beta*0.25*(
            np.sum(np.trace(self.DeltaBlock[i, i]@(G_w_flt_up.reshape(self.n_iw, dnwa , dnwa))@\
                            self.DeltaBlock[j ,j]@(G_w_flt_dn.reshape(self.n_iw, dnwa , dnwa)), axis1=1, axis2=2)
                    ))
        
        J = J*2./(self.S**2)
        if verb:
            print(f'J{i}{j}= {1000*J:.5f} meV')
        return J


    @abstractmethod
    def get_kqpt(self, kpt, q):
        '''
        shift flat indice (kpt) on target vector q 
        '''
        pass

    def getJqpath(self, i, j, kpath):
        '''
        i, j - sites
        kpath - integer kpath
        '''
        Jqs = []
        for q in tqdm(kpath):
            val = 0
            for kpt in range(self.nkpt):
                val = val - 1.0/self.beta*0.25*self.dSarea*np.sum(np.trace(self.DeltaBlock[i, i]@self.Giws_dnBlock[i, j, self.get_kqpt(kpt, q) ]\
                                                                    @self.DeltaBlock[j, j]@self.Giws_upBlock[j, i, kpt], axis1=1, axis2=2), axis=0)
            Jqs.append(val)
        Jqs = np.array(Jqs)*2./(self.S**2) - self.Jself[i]*(i==j)
        return Jqs
        # assert np.allclose(1000*np.sum(Jqs - Jself[i])*dSarea, 0)



    def getJRpath(self, R_path, i, j, verb=True, Jqs_dense=None):
        '''
        gets J(R) from J(q) along R_path
        '''

        if Jqs_dense is None:
            # making dense J(q) mesh
            Jqs_dense = self.getJqpath(self, i, j, self.kMesh_int)
            
            # Jqs = []
            # for q in tqdm(self.kMesh_int):
            #     val = 0
            #     for kpt in range(self.nkpt):
            #         val = val - 1.0/self.beta*0.25*self.dSarea*np.sum(np.trace(self.DeltaBlock[i, i]@self.Giws_dnBlock[i, j, self.get_kqpt(kpt, q) ]\
            #                                                             @self.DeltaBlock[j, j]@self.Giws_upBlock[j, i, kpt], axis1=1, axis2=2), axis=0)
            #     Jqs.append(val)*2./(self.S**2)
            # Jqs = np.array(Jqs)
            if  i == j : assert np.abs(np.sum(Jqs_dense - self.Jself[i])*self.dSarea) < 1e-9, 'smth goes wrong'
        

        J_real = []
        for r in tqdm(R_path):
            J_real.append( sum( np.exp(-2.0*np.pi*1.j*  self.kMesh_frac[i]@(r))*Jqs_dense[i] for i in range(len(self.kMesh_frac))) / self.nkpt )
            if verb:
                print(f'J{i}{j}= {1000*J_real[-1]:.5f} meV')
        J_real = np.array(J_real)

        return J_real

