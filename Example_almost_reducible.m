clear all; close all; clc
rng(12)

m = 5;
P1 = rand(m,m);
P1 = diag(sum(P1,2))\P1;
P2 = rand(m,m);
P2 = diag(sum(P2,2))\P2;

Pf = blkdiag(P1,P2);
Pf(1,end) = (1e-5)*rand; %Esempio problematico?
Pf(end,1) = (1e-5)*rand; %Esempio problematico?

% Pf(1,end) = rand;
% Pf(end,1) = rand;

A = diag(sum(Pf,2))\Pf;

[pi,~] = eigs(Pf',1,'largestabs');
pi = abs(pi); pi = pi/sum(pi);
pv = pi.^(1/2); Dv = diag(pv);

n = size(A,1);

TT = eye(n) + A; %+ Pf^2;
S = double(TT > 0);

% %Riemannian optimization
[Xs, xcost] = optimizeriemm(A,pi,S);

%Additional checks on the Reversibility - RIEMANNIAN CASE
norm(diag(pi)*Xs - Xs'*diag(pi))
norm(Xs*ones(n,1) - ones(n,1))
norm(pi'*Xs - pi')

fprintf("||Xs - A|| = %e\n", norm(Xs - A,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*Xs*(Dv^-1) + pv*pv')\eye(n)), trace((eye(n) - Dv*A*(Dv^-1) + pv*pv')\eye(n)));
 
% %Initialize the manifold
% M = multinomialsparsesymmetricfixedfactory(pv,S);
% 
% Bhat = diag(pv.^(-1))*Pf*diag(pv); %Transform the matrix
% B0 = Pf;
% 
% %Initialization of the problem
% problem.M = M;
% problem.cost = @(X) trace((eye(n) - X + pv*pv')\eye(n)) + 0.5*norm(diag(pv.^(-1))*X*diag(pv) - B0,'fro')^2;
% problem.egrad = @(X) ((eye(n) - X + pv*pv')^2\eye(n))' + (diag(pi.^(-1))*X*diag(pi) - diag(pv.^(-1))*B0*diag(pv));
% 
% options.tolgradnorm = 1e-10;
% options.verbosity = 0;
% 
% [X1, xcost, info, options] = conjugategradient(problem, [], options);
% [X1, xcost, info, options] = steepestdescent(problem, [], options);
% Xs = diag(pv.^(-1))*X1*diag(pv);

[Delta,varargout] = optimizekemeny(A, 'SparsePreserving', pi);

%Additional checks on the Reversibility - EUCLIDEAN CASE
norm(diag(pi)*(A+Delta) - (A+Delta)'*diag(pi))
norm((A+Delta)*ones(n,1) - ones(n,1))
norm(pi'*(A+Delta) - pi')

fprintf("||(A+Delta) - A|| = %e\n", norm((A+Delta) - A,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*(A+Delta)*(Dv^-1) + pv*pv')\eye(n)), trace((eye(n) - Dv*A*(Dv^-1) + pv*pv')\eye(n)));