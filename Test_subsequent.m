%Test for subsequent minimization with increasing tolerances
clear all; close all; clc
rng(105)
n = 50;

%Initialize the stochastic matrix and its stationary distribution
pi = rand(n,1);
pi = abs(pi); pi = pi/sum(pi);
pv = pi.^(1/2);
Dv = diag(pv);

%Create a sparsity pattern
S = triu(randi([0 1], n, n), 1);
S = S + S' + eye(n);

N = multinomialsparsesymmetricfixedfactory(pv,S);
A = N.rand(); %symmetric and fixed eigenvector pv
A =  diag(pv.^(-1))*A*diag(pv); %this is reversible (stochastic)

[Xs,cost] = optimizeriemm(A,pi,S);

%Additional checks on the Reversibility - RIEMANNIAN CASE
norm(diag(pi)*Xs - Xs'*diag(pi))
norm(Xs*ones(n,1) - ones(n,1))
norm(pi'*Xs - pi')

fprintf("||Xs - A|| = %e\n", norm(Xs - A,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*Xs*(Dv^-1) + pv*pv')\eye(n)), trace((eye(n) - Dv*A*(Dv^-1) + pv*pv')\eye(n)));

[Delta,varargout] = optimizekemeny(A, 'SparsePreserving', pi);

%Additional checks on the Reversibility - EUCLIDEAN CASE
norm(diag(pi)*(A+Delta) - (A+Delta)'*diag(pi))
norm((A+Delta)*ones(n,1) - ones(n,1))
norm(pi'*(A+Delta) - pi')

fprintf("||(A+Delta) - A|| = %e\n", norm((A+Delta) - A,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*(A+Delta)*(Dv^-1) + pv*pv')\eye(n)), trace((eye(n) - Dv*A*(Dv^-1) + pv*pv')\eye(n)));