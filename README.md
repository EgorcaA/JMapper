<!-- ABOUT THE PROJECT -->
## About The Project

This tool is designed to help calculate Heisenberg exchange parameters in Green's functions formalism and visualize the results.
1ML notebook contains calculations of CrTe2 monolayer, 2ML - bilayer with 2 magnetic atoms in cell. 
Use `exchange_GG_2atoms.py` file to adapt the program for complex hamiltonian structure.  

<!-- USAGE -->
## USAGE

To get exchange parameters one should
* run scf and nscf calculations in DFT package
* obtain wannier hamiltonian in wanner90 standard hr.dat format for both spin components
* use notebooks to get the results!

### Prerequisites

* triqs
  ```sh
  mamba install triqs
  ```
  *numpy
  *matplotlib
  *tqdm


Tool to help extract exchange integrals from DFT 
<!-- CONTACT -->
## Contact

Egor Agapov -  agapov.em@phystech.edu
Project Link: [https://github.com/EgorcaA/exchanges_DFT](https://github.com/EgorcaA/exchanges_DFT)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

