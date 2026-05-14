% ============================================================
% SDP for Optimizing the Kemeny Constant of a Reversible MC
%
% Inputs:
%   n    : number of states
%   pi   : stationary distribution column vector (n x 1)
%   E    : adjacency matrix of graph G (n x n)
%          E(i,j)=1 if edge exists, 0 otherwise
%
% Outputs:
%   Popt : optimal transition matrix
%   Xopt : optimal SDP matrix variable
% ============================================================

clear; clc; close all;

n = 30;
%Initialize the stochastic matrix and its stationary distribution
ps = rand(n,1);
ps = abs(ps); pi = ps/sum(ps);
pv = pi.^(1/2);
Dv = diag(pv);

%Create a sparsity pattern
E = triu(randi([0 1], n, n), 1);
E = E + E' + eye(n);

N = multinomialsparsesymmetricfixedfactory(pv,E);
A = N.rand(); %symmetric and fixed eigenvector pv
A =  diag(pv.^(-1))*A*diag(pv); %this is reversible (stochastic)

old_kem = trace((eye(n) - Dv*A*(Dv^-1) + pv*pv')\eye(n));

tic;
[Xs_cg,cost_cg] = optimizeriemm(A,pi,E,1);
time_cg = toc;

new_kem_riemann = trace((eye(n) - Dv*Xs_cg*(Dv^-1) + pv*pv')\eye(n));

fprintf("---- Riemannian ------\n")
fprintf("Original Kemeny Constant: %f\n",old_kem);
fprintf("New Kemeny's Constant (Riemannian): %f obtained in %e s\n",new_kem_riemann,time_cg);
fprintf("Relative Distance from the original chain: %e\n",norm(Xs_cg - A,"fro")/norm(A,"fro"));
fprintf("Reversibility: %e\n", norm(diag(pi)*Xs_cg - Xs_cg'*diag(pi),inf));
fprintf("Stochasticity: %e\n", norm(Xs_cg*ones(n,1) - ones(n,1),inf));
fprintf("Stationarity: %e\n", norm(pi'*Xs_cg - pi',inf));

% diagonal matrix Π and vector q = sqrt(pi)
Pi      = diag(pi);
Pi_half = diag(sqrt(pi));
Pi_mh   = diag(1./sqrt(pi));
q = sqrt(pi);



fprintf("\n\n\n---- SDP ------\n\n\n")
tic;
cvx_begin sdp

variable P(n,n)
variable X(n,n) symmetric

% Objective
minimize( trace(X) )

subject to

% ----------------------------------------------------
% LMI constraint:
%
% [ I - Π^{1/2} P Π^{-1/2} + q q^T    I ;
%                  I                  X ] >= 0
% ----------------------------------------------------
A = eye(n) - Pi_half * P * Pi_mh + q*q';

[A, eye(n);
    eye(n), X] >= 0;

% Row stochastic constraints
sum(P,2) == ones(n,1);

% Reversibility constraints:
% pi_i p_ij = pi_j p_ji
for i = 1:n
    for j = 1:n
        if E(i,j) == 1
            pi(i)*P(i,j) == pi(j)*P(j,i);
        end
    end
end

% Edge constraints and bounds
for i = 1:n
    for j = 1:n

        if E(i,j) == 1
            0 <= P(i,j) <= 1;
        else
            P(i,j) == 0;
        end

    end
end

cvx_end
time_sdp = toc;

% Optimal solutions
Popt = P;
Xopt = X;

kem_sdp = trace((eye(n) - Dv*Popt*(Dv^-1) + pv*pv')\eye(n));

fprintf("Kemeny constant from SDP: %e obtained in %e s\n",kem_sdp,time_sdp);
fprintf("Relative Distance from the original chain: %e\n",norm(Popt - A,"fro")/norm(A,"fro"));
fprintf("Reversibility: %e\n", norm(diag(pi)*Popt - Popt'*diag(pi),inf));
fprintf("Stochasticity: %e\n", norm(Popt*ones(n,1) - ones(n,1),inf));
fprintf("Stationarity: %e\n", norm(pi'*Popt - pi',inf));