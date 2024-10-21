# Utilities

This folder contains some additional functions needed to generate examples 
or build formulations for optimization problems.

- `T = build_sparse_T(n)`: This function builds a sparse matrix T such that
   `T*v = w` where `v = reshape(P', n^2, 1)` and `w = P(:)`;
- `Q = reversible_markov_chain(P, pi, method)`: takes as input an ergodic 
   stochastic matrix P and its stationary vector pi and produces a 
   stochastic matrix Q of a reversible Markov chain with stationary 
   vector pi. It uses either `"barker"` or `'metropolis'` method.
- `proj = pattern_projector(S)` build the matrix projecting a generic vector
   with n^2 entries onto the vector with nnz(S) entries corresponding to the
   linear indexing of the nonzero entries of S.  