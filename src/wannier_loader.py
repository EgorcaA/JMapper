import numpy as np
import pickle 
from  tqdm import tqdm


Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr


class Wannier_loader():
    '''
    basic class for import wannier90 hamiltonians
    '''
    def __init__(self, dir):
        '''
        directory of wannier hamiltonian(s)
        '''
        self.directory = dir 
        self.nwa = 0
        self.load_wannier90()
    

    def load_kpath(self, path):
        '''
        reads kpath for plotting interpolated wannier band structure
        in format: kpt_name kx_frac ky_frac kz_frac dist
        
        example: 

        G 0.00000000  0.00000000 0.00000000  	 0.00000000 
        . 0.00000000  0.10000000 0.00000000  	 0.11547369 
        ...
        
        '''
        k_path = []
        kpath_dists = []

        with open(path) as f:
            for line in f:
                kpts_string = line.split()
                k_path.append(np.array([
                    float(kpts_string[1]), float(kpts_string[2]), float(kpts_string[3])
                ]))
                kpath_dists.append(float(kpts_string[4]))

        self.k_path_qe = np.array(k_path)   
        self.kpath_dists_qe = np.array(kpath_dists)   


    def load_util(self, filename):
            '''
            loads standart wannier90 hr file
            '''
            hr = 0
            R_coords = []
            R_weights = []
            flag3D = 0
            with open(self.directory  + filename + '.dat') as f:
                    f.readline()
                    
                    nwa = int(f.readline().strip('\n'))
                    print("nwa ", nwa)
                    self.nwa = nwa
                    Rpts = int(f.readline().strip('\n'))
                    print("Rpts", Rpts)
                    i=1
                    hr = np.zeros((nwa, nwa, Rpts), dtype=complex)

                    R_ind = -1
                    line_ind = 0
                    for line in f:
                        
                        if i< Rpts/15+1:
                            R_weights +=  [ int(x) for x in line.split() if x.isnumeric() ]
                            i+=1
                        else:

                            hr_string = line.split()
                            
                            if line_ind % nwa**2 == 0:
                                if float(hr_string[2]) != 0: flag3D = 1 
                                R_coords.append([float(hr_string[0]), float(hr_string[1]), float(hr_string[2])]) 
                                R_ind += 1
                            hr[int(hr_string[3])-1, int(hr_string[4])-1, R_ind] = float(hr_string[5])+ 1j*float(hr_string[6])
                            
                            line_ind +=1 
            self.nD = flag3D + 2

            print(f'we have {self.nD}D hamiltonian')
            return R_coords, hr

    

class Wannier_loader_FM(Wannier_loader):
    '''
    class for import wannier90 ferromanetic hamiltonians 
    (with 2 spins, up and down)
    '''
    def __init__(self,  dir):
        super().__init__( dir) 

    def load_wannier90(self):
        '''
        default names of spin up and spin down hamiltonians 
        are hrup.dat and hrdn.dat resp.
        '''
        R_coords, hr_up = self.load_util('hrup')
        _, hr_dn = self.load_util('hrdn')
        self.complex_hr = np.transpose(np.array([hr_up, hr_dn]), (1,2,3,0))
        self.R_coords = R_coords
        
    def save_hk_pickle(self):
        with open(self.directory + '/hk_dense.pickle', 'wb') as f:
            pickle.dump(self.hks_spins, f)

    def load_hk_pickle(self):
        with open(self.directory + '/hk_dense.pickle', 'rb') as f:
            self.hks_spins = pickle.load(f)


    def get_dense_hk(self, nkpt=10):
        '''
        evaluates h_mn(k) on dense kgrid 
        nkpt is number of points in each direction 
        '''
        spins = [0, 1]
        if self.nD == 3: 
            kpoints_adj_serial = np.mgrid[0:1:1.0/nkpt, 0:1:1.0/nkpt, 0:1:1.0/nkpt].reshape(3,-1).T

            hks_spins = []
            for spin in spins:
                hks = []
                for k in tqdm(kpoints_adj_serial):
                    hk = np.sum( [np.exp(2*np.pi*1.j* np.dot(k, R) )*self.complex_hr[:, :, R_ind, spin] 
                                for R_ind, R in enumerate(self.R_coords)], axis=0 )
                    hk = np.array(hk).T
                    hks.append(hk)
                hks_spins.append(hks)
            self.hks_spins = np.transpose( np.array(hks_spins) , (2,3, 1,0))
            
        else:
            kpoints_adj_serial = np.mgrid[0:1:1.0/nkpt, 0:1:1.0/nkpt].reshape(2,-1).T

            hks_spins = []
            for spin in spins:
                hks = []
                for k in tqdm(kpoints_adj_serial):
                    hk = np.sum( [np.exp(2*np.pi*1.j* np.dot(k, R[:2]) )*self.complex_hr[:, :, R_ind, spin] 
                                for R_ind, R in enumerate(self.R_coords)], axis=0 )
                    hk = np.array(hk).T
                    hks.append(hk)
                hks_spins.append(hks)
            self.hks_spins = np.transpose( np.array(hks_spins) , (2,3, 1,0))

        
        self.kpoints_adj_serial = kpoints_adj_serial

    def get_wannier_BS(self, spin=0): 
        '''
        plots wannier BS on predefined kpath
        '''
        band_str = []
        self.hks_bs = []
        for k in tqdm(self.k_path_qe):
            hk = np.sum( [np.exp(2*np.pi*1.j* np.dot(k, R) )*self.complex_hr[:, :, R_ind, spin] 
                          for R_ind, R in enumerate(self.R_coords)], axis=0 )
            self.hks_bs.append(hk)
            band_str.append(np.sort(np.real(np.linalg.eig(hk)[0])))
        band_str = np.array(band_str)
        return band_str

    
class Wannier_loader_PM(Wannier_loader):
    '''
    class for import wannier90 paramagnetic hamiltonians
    '''
    def __init__(self,  dir):
        super().__init__( dir) 

    def load_wannier90(self):
        
        R_coords, hr = self.load_util('hr')
        self.complex_hr = hr
        self.R_coords = R_coords
        

    def get_dense_hk(self):
        '''
        evaluates h_mn(k) on dense kgrid 
        nkpt is number of points in each direction 
        '''
        kpoints_adj_serial = np.mgrid[0:1:1.0/10., 0:1:1.0/10., 0:1:1.0/10.].reshape(3,-1).T

        hks = []
        for k in tqdm(kpoints_adj_serial):
            hk = np.sum( [np.exp(2*np.pi*1.j* np.dot(k, R) )*self.complex_hr[:, :, R_ind] 
                        for R_ind, R in enumerate(self.R_coords)], axis=0 )
            hk = np.array(hk).T
            hks.append(hk)

        self.hks = np.transpose( np.array(hks) , (1,2, 0))
        self.kpoints_adj_serial = kpoints_adj_serial


    def get_wannier_BS(self): 
        '''
        plots wannier BS on predefined kpath
        '''
        band_str = []
        self.hks_bs = []
        for k in tqdm(self.k_path_qe):
            hk = np.sum( [np.exp(2*np.pi*1.j* np.dot(k, R) )*self.complex_hr[:, :, R_ind] 
                          for R_ind, R in enumerate(self.R_coords)], axis=0 )
            self.hks_bs.append(hk)
            band_str.append(np.sort(np.real(np.linalg.eig(hk)[0])))
        band_str = np.array(band_str)
        return band_str
