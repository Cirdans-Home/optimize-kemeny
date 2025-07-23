%Script for minimizing the Kemeny constant over reversible matrices

%clear all; close all; clc
%rng(105)
%n = 10;

% %initialize the steady vector and its square root (entry-wise)
pi = rand(n,1); pi = pi/norm(pi,1);
pv = pi.^(1/2);

%Create a sparsity pattern
S = triu(randi([0 1], n, n), 1);
S = S + S' + eye(n);

%Initialize the manifold and an initial point
M = multinomialsparsesymmetricfixedfactory(pv,S);

delta = 1e-5;
E = delta*rand(n);
Dv = diag(pv);

B = M.rand();

Bhat = diag(pv.^(-1))*B*diag(pv); %This is reversible, Bhat*e = e, pi'*Bhat = pi'
B0 = Bhat; %it this way we start from a reversible matrix

%Initialization of the problem
problem.M = M;
problem.cost = @(X) trace((eye(n) - X + pv*pv')\eye(n)) + 0.5*norm(diag(pv.^(-1))*X*diag(pv) - B0,'fro')^2;
problem.egrad = @(X) ((eye(n) - X + pv*pv')^2\eye(n))' + (diag(pi.^(-1))*X*diag(pi) - diag(pv.^(-1))*B0*diag(pv));
    
options.tolgradnorm = 1e-12;
options.linesearch = @linesearch_adaptive;
options.verbosity = 0;

[X1, xcost, info, options] = conjugategradient(problem, B, options);

Xs = diag(pv.^(-1))*X1*diag(pv);