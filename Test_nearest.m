%Script per un test iniziale:
% questo dovrebbe calcolare nearest sparse reversible,
% però possiamo usarlo per testare la manifold

%MIRYAM: Mi sembra che con un metodo del primo ordine si faccia un po' fatica...
%forse conviene fare i conti per l'Hessiano

clear all; close all; clc
rng(105)
n = 20;

%initialize the steady vector and its square root (entry-wise)
pi = rand(n,1); pi = pi/norm(pi,1);
pv = pi.^(1/2);

S = triu(randi([0 1], n, n), 1);
S = S + S' + eye(n);

M = multinomialsparsesymmetricfixedfactory(pv,S);

delta = 1e-5;
E = delta*rand(n);
Dv = diag(pv);

B = M.rand();

Bhat = diag(pv.^(-1))*B*diag(pv); %This is reversible, Bhat*e = e, pi'*Bhat = pi'
B0 = Bhat + E; %perturbation of Bhat
%B0 = Bhat + S.*E; %perturbation of Bhat Structured

%Initialization of the problem
problem.M = M;
%problem.cost = @(X) cnormsqfro(diag(pv.^(-1))*X*diag(pv) - B0);
%problem = manoptAD(problem);

problem.cost = @(X) 0.5*norm(diag(pv.^(-1))*X*diag(pv) - B0,'fro')^2;
problem.egrad = @(X) (diag(pi.^(-1))*X*diag(pi) - diag(pv.^(-1))*B0*diag(pv));

checkgradient(problem)

options.tolgradnorm = 1e-10;
options.verbosity = 0;

[X1, xcost, info, options] = conjugategradient(problem,[], options);
%[X1, xcost, info, options] = trustregions(problem, [], options);

Xs = diag(pv.^(-1))*X1*diag(pv);