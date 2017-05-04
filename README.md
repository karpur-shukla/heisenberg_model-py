# heisenberg_model-py
This is a repository for all of my code related to the Heisenberg model in statistical mechanics, as implemented in Python 2.7.12 (using NumPy, SciPy, and matplotlib). This repository contains the following files:
## Shukla - Ising Model 0NN.py
This is a two-dimensional (N-by-M) Ising model with no nearest-neighbour interactions. Since we have no nearest-neighbour interactions, the partition function can be directly factorised, and this system is equivalent to NM copies of a single spin.
## Shukla - Ising Model 2D.py
This is a two-dimensional (N-by-M) Ising model with first-nearest-neighbour interactions. The script permits anisotropy by specifying distinct values for the x-directional and y-directional couplings.
## Shukla - Ising Model 2D Histogram.py
This is a two-dimensional (N-by-M) Ising model with first-nearest-neighbour interactions, which provides histograms of the resulting average total magnetisations <M> and average total energies <E>. This is part of an attempt to recreate the weighted histogram analysis method (WHAM) seen in A. Ferrenberg & R. Swendsen, Phys. Rev. Lett. 61, 23 (1988) and A. Ferrenberg & R. Swendsen, Phys. Rev. Lett. 63, 12 (1989).
