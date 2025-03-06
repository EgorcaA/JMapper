import numpy as np
import numpy.linalg as LA


Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr


class NeighboursFinder2D():

    def __init__(self, qeAnalyse):

        # cartisian coordinates (A)
        self.celldims = qeAnalyse.acell
        self.names = qeAnalyse.pos[0]

        # fractional coordinates
        self.positions = np.array(qeAnalyse.pos[1])*Bohr2Ang @ LA.inv(self.celldims)
        self.positions =self.positions[:, :2]

    def find(self, i, j, num_orders=5, num_vert=10, precision=2):
        print(self.names[i], self.names[j])

        init_shift = self.positions[i] - self.positions[j]
        # print(init_shift)
        vert_J = {}
        for i_index in range(-num_vert, num_vert):
            for j_index in range(-num_vert, num_vert):
                pair = [i_index, j_index ]
                dist = np.round(np.linalg.norm((pair[0] + init_shift[0])*self.celldims[0] + (pair[1] + init_shift[1])*self.celldims[1]), precision)
                # print(pair, dist)
                if dist not in vert_J: 
                    vert_J[dist]=[pair]
                else:
                    vert_J[dist].append(pair)
                # print(pair, dist)
        sorted_vert_J = dict(list(sorted(vert_J.items()))[:num_orders]) #dict(sorted(vert_J.items()))
        # for i, item in enumerate(sorted_vert_J.items()):
        #     print(item[1])
        #     if i > 5: break

        return sorted_vert_J
    

class NeighboursFinder3D():

    def __init__(self, qeAnalyse):
        
        # cartisian coordinates (A)
        self.celldims = qeAnalyse.acell
        self.names = qeAnalyse.pos[0] # names of atoms

        # fractional coordinates
        self.positions = np.array(qeAnalyse.pos[1])*Bohr2Ang  @ LA.inv(self.celldims)

    def find(self, i, j, num_orders=5, num_vert=10, precision=2):
        print(self.names[i], self.names[j])

        init_shift = self.positions[i] - self.positions[j]
        vert_J = {}
        for i_index in range(-num_vert, num_vert):
            for j_index in range(-num_vert, num_vert):
                for k_index in range(-num_vert, num_vert):
                    pair = [i_index, j_index, k_index ]
                    dist = np.round(np.linalg.norm( (pair[0] + init_shift[0])*self.celldims[0] + (pair[1] + init_shift[1])*self.celldims[1] + (pair[2] + init_shift[2])*self.celldims[2] ), precision)
                    # print(pair, dist)
                    if dist not in vert_J: 
                        vert_J[dist]=[pair]
                    else:
                        vert_J[dist].append(pair)

        sorted_vert_J = dict(list(sorted(vert_J.items()))[:num_orders]) 

        return sorted_vert_J