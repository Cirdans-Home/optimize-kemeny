%Test with respect to the dimension of the problem
% Test with prescribed sparsity pattern, and fixed entries
% for synthetic stochastic matrices
% Comparison among bb, cg and Euclidean optimizer
clear all; close all; clc
rng (19)

size_m = 60;
tests = 25; 
dim = size_m.*ones(1,tests);

index_nan=[];

l = length(dim);

rev = zeros(3,l);
stat = zeros(3,l);
stoc = zeros(3,l);
dist_rel = zeros(3,l);

old_kem = zeros(1,l);
new_kem = zeros(3,l);

times = zeros(3,l);

for i = 1:l

    n = dim(i);
    %Initialize the stochastic matrix and its stationary distribution
    ps = rand(n,1);
    ps = abs(ps); ps = ps/sum(ps);
    pv = ps.^(1/2);
    Dv = diag(pv);
    
    %Create a sparsity pattern and fixed entries
    S = triu(randi([0 1], n, n), 1);
    T = max(triu(randi([0 1], n, n), 1) - S, 0);
    S = S + S' + eye(n);
   
    W = multinomialsymmetricfixedfactory(pv);
    R = diag(pv.^(-1))*W.rand()*diag(pv);
    P = R.*(T+T');

    N = multinomialsymmetricfixedentriesfactory(pv,S,P);
    A = N.rand(); %symmetric and fixed eigenvector pv, with sparsity pattern S and fixed entries P
    A = diag(pv.^(-1))*A*diag(pv); %this is reversible (stochastic)

    old_kem(i) = trace((eye(n) - Dv*A*(Dv^-1) + pv*pv')\eye(n));
    
    tic;
    [Xs_cg,cost_cg] = optimizeriemm(A,ps,S,1,P);
    time_cg = toc;
    %Additional checks on the Reversibility - RIEMANNIAN CASE - CG
    rev(1,i) = norm(diag(ps)*Xs_cg - Xs_cg'*diag(ps),inf);
    stoc(1,i) = norm(Xs_cg*ones(n,1) - ones(n,1),inf);
    stat(1,i) = norm(ps'*Xs_cg - ps',inf);
    dist_rel(1,i) =  norm(Xs_cg - A,"fro")/norm(A,"fro");
    new_kem(1,i) = trace((eye(n) - Dv*Xs_cg*(Dv^-1) + pv*pv')\eye(n));
    times(1,i) =  time_cg;

    %check-FIXED ELEMENTS
     if (max(max(abs(full(Xs_cg.*spones(P))-P)))>1e-15)
            error('elements not fixed ');
     end

    tic;
    [Xs_bb,cost_bb] = optimizeriemm(A,ps,S,2,P);
    time_bb = toc;
    %Additional checks on the Reversibility - RIEMANNIAN CASE - BB
    rev(2,i) = norm(diag(ps)*Xs_bb - Xs_bb'*diag(ps),inf);
    stoc(2,i) = norm(Xs_bb*ones(n,1) - ones(n,1),inf);
    stat(2,i) = norm(ps'*Xs_bb - ps',inf);
    dist_rel(2,i) =  norm(Xs_bb - A,"fro")/norm(A,"fro");
    new_kem(2,i) = trace((eye(n) - Dv*Xs_bb*(Dv^-1) + pv*pv')\eye(n));
    times(2,i) =  time_bb;


    %check-FIXED ELEMENTS
    if (max(max(abs(full(Xs_bb.*spones(P))-P)))>1e-15)
        error('elements not fixed ');
    end

    %check Nan in BB
      if (isnan(Xs_bb))
           index_nan = [index_nan; i];
      end

    tic;
    [Delta,varargout] = optimizekemeny(A, 'SparsePreserving', ps, spones(S));
    time_eu = toc;
    %Additional checks on the Reversibility - EUCLIDEAN CASE
    rev(3,i) = norm(diag(ps)*(A+Delta) - (A+Delta)'*diag(ps),inf);
    stoc(3,i) = norm((A+Delta)*ones(n,1) - ones(n,1),inf);
    stat(3,i) = norm(ps'*(A+Delta) - ps',inf);
    dist_rel(3,i) =  norm(Delta,"fro")/norm(A,"fro");
    new_kem(3,i) = trace((eye(n) - Dv*(A+Delta)*(Dv^-1) + pv*pv')\eye(n));
    times(3,i) =  time_eu;


    %check-FIXED ELEMENTS
    if (max(max(abs(full((A+Delta).*spones(P))-P)))>1e-15)
        error('elements not fixed ');
    end

end

save('fixed_60.mat', 'rev', 'stat', 'stoc', 'dist_rel', 'old_kem', 'new_kem','times')