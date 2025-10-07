%Script-- Testing the code on an almost reducible matrix
%with high Kemeny's constant 
clear all; close all; clc
rng(12)

m = 25;
P1 = rand(m,m);
P1 = diag(sum(P1,2))\P1;
P2 = rand(m,m);
P2 = diag(sum(P2,2))\P2;

Pf = blkdiag(P1,P2);
Pf(1,end) = (1e-5)*rand; %
Pf(end,1) = (1e-5)*rand; %

P = diag(sum(Pf,2))\Pf;

S = full(spones(P));

n = 2*m;
ps = rand(n,1); ps = abs(ps);
ps = ps/sum(ps);

A = zeros(n);

    for i = 1:n
        for j = 1:n
            if i ~= j && P(i,j) > 0
                % Using Metropolis-Hastings formula
                mh_ratio = (ps(j) * P(j,i)) / (ps(i) * P(i,j));
                alpha = min(1, mh_ratio);
                A(i,j) = P(i,j) * alpha;
            end
        end
    end

    %Make the matrix stochastic
    for i = 1:n
        A(i,i) = 1 - sum(A(i,[1:i-1, i+1:end]));
    end

 pv = ps.^(1/2); 
 Dv = diag(pv);

%%Riemannian optimization
tic;
[Xs, xcost] = optimizeriemm(A,ps,S);
time_riemm = toc;

%Additional checks on the Reversibility - RIEMANNIAN CASE
norm(diag(ps)*Xs - Xs'*diag(ps))
norm(Xs*ones(n,1) - ones(n,1))
norm(ps'*Xs - ps')

fprintf("||Xs - A|| = %e\n", norm(Xs - A,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*Xs*(Dv^-1) + pv*pv')\eye(n))-1, trace((eye(n) - Dv*A*(Dv^-1) + pv*pv')\eye(n))-1);

tic;
[Delta,varargout] = optimizekemeny(A, 'SparsePreserving', ps);
time_fmin = toc;

%Additional checks on the Reversibility - EUCLIDEAN CASE
norm(diag(ps)*(A+Delta) - (A+Delta)'*diag(ps))
norm((A+Delta)*ones(n,1) - ones(n,1))
norm(ps'*(A+Delta) - ps')

fprintf("||(A+Delta) - A|| = %e\n", norm((A+Delta) - A,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*(A+Delta)*(Dv^-1) + pv*pv')\eye(n))-1, trace((eye(n) - Dv*A*(Dv^-1) + pv*pv')\eye(n))-1);