# Minimize Kemeny’s Constant

The algorithms implemented here solve the problem of finding a (structured) 
perturbation matrix to be added to a given stochastic matrix in such a way 
that the Kemeny's constant of the resulting matrix is ​​smaller than that of 
the starting matrix. That is, that the chain connection is improved.

## Collaborators

- Fabio Durastante [📧](mailto:fabio.durastante@unipi.it) [🌐](https://fdurastante.github.io/)
- Miryam Gnazzo [📧](mailto:miryam.gnazzo@dm.unipi.it) 
- Beatrice Meini [📧](mailto:beatrice.meini@unipi.it) [🌐](https://people.dm.unipi.it/meini/)

## External code

This project uses **[Manopt](https://www.manopt.org/)**, the MATLAB toolbox 
for optimization on manifolds, included as a Git submodule.

### 🔽 Cloning with Manopt

To clone the repository along with the Manopt submodule, use:

```bash
git clone --recurse-submodules git@github.com:Cirdans-Home/optimize-kemeny.git
```

If you've already cloned the repository without submodules, run:

```bash
cd optimize-kemeny
git submodule update --init --recursive
```

### 🛠 Usage in MATLAB

After cloning, you can set up Manopt in MATLAB with:

```matlab
addpath('manopt');
importmanopt;   % optional, if available
checkinstall;   % optional, to verify the installation
```

You are now ready to use the functions in our code which rely on Manopt

### How to cite

If you use our code together with any version of Manopt please cite
```bibtex
@Article{manopt,
    author  = {Boumal, N. and Mishra, B. and Absil, P.-A. and Sepulchre, R.},
    journal = {Journal of Machine Learning Research},
    title   = {{M}anopt, a {M}atlab Toolbox for Optimization on Manifolds},
    year    = {2014},
    number  = {42},
    pages   = {1455--1459},
    volume  = {15},
    url     = {https://www.manopt.org}
}
```

