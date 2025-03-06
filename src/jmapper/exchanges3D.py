from .base_exchanges import Exchanges # Relative import

from triqs.gf import Gf, MeshImFreq, iOmega_n, inverse
import numpy as np


class Exchanges3D(Exchanges):
    def __init__(self, Giws_upBlock, Giws_dnBlock, DeltaBlock, beta, nkpt_dir, S):
        super().__init__( Giws_upBlock, Giws_dnBlock, DeltaBlock, beta, S)

        self.nkpt_dir = nkpt_dir
        
        self.kMesh_frac = np.mgrid[0:1:1.0/self.nkpt_dir, 0:1:1.0/self.nkpt_dir, 0:1:1.0/self.nkpt_dir].reshape(3,-1).T #3D
        self.kMesh_int = np.mgrid[0:self.nkpt_dir:1, 0:self.nkpt_dir:1, 0:self.nkpt_dir:1].reshape(3,-1).T #3D
        
        assert self.nkpt_dir**3 == self.nkpt, "kpoint num is strange"
        # print("hh")
        Jselfs = []
        for site in range(self.num_sites):
            Jselfs.append(self.calc_J(np.array([0, 0, 0]), site, site, verb=True)) # self interaction
        self.Jself = np.array(Jselfs)


    def get_kqpt(self, kpt, q):
        kx = kpt//self.nkpt_dir**2
        ky = (kpt - kx*(self.nkpt_dir**2))//self.nkpt_dir
        kz = kpt - ky*self.nkpt_dir - kx*(self.nkpt_dir**2)
        k = np.array([kx, ky, kz])
        kq = np.mod(k+q, self.nkpt_dir)
        kqpt = kq[0]*(self.nkpt_dir**2) + kq[1]*self.nkpt_dir + kq[2]
        return kqpt 

    





