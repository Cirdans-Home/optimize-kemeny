%Script for a graph example
clear all; close all; clc

load('voltage_adjacencies_average_2.mat')

%SWITZERLAND -- Irreducible Example
A = Switzerland;


% % Other countries -- Reducible examples
%  AA = Netherlands;
%  G = graph(AA);
%  [bins,binsize] = G.conncomp("OutputForm","vector");
% 
%  [max_size,max_i] = max(binsize);
% 
%  A = AA(bins == max_i,bins == max_i);

n = size(A,1);

d = sum(A,2);
P = diag(d.^(-1))*A;

P = full(P);

%stationary distribution
ps = abs(d);
ps = ps/sum(ps);

pv = ps.^(1/2);
Dv = diag(pv);

%Choosing the pattern
S = full(spones(P+eye(n)));

% %Euclidean optimization
tic;
[Delta,varargout] = optimizekemeny(P, 'SparsePreserving', ps, spones(S));
time_fmin = toc;

%Additional checks on the Reversibility - RIEMANNIAN CASE
norm(diag(ps)*(P+Delta) - (P+Delta)'*diag(ps),'inf')
norm((P+Delta)*ones(n,1) - ones(n,1),'inf')
norm(ps'*(P+Delta) - ps','inf')

fprintf("||Xs - A|| = %e\n", norm(Delta,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*(P+Delta)*(Dv^-1) + pv*pv')\eye(n))-1, trace((eye(n) - Dv*P*(Dv^-1) + pv*pv')\eye(n))-1);

A_eucl = diag(d)*(P+Delta);

% %Riemannian optimization
tic;
[Xs, xcost] = optimizeriemm(P,ps,S);
time_riemm = toc;

%Additional checks on the Reversibility - RIEMANNIAN CASE
norm(diag(ps)*Xs - Xs'*diag(ps),'inf')
norm(Xs*ones(n,1) - ones(n,1),'inf')
norm(ps'*Xs - ps','inf')

fprintf("||Xs - P|| = %e\n", norm(Xs - P,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*Xs*(Dv^-1) + pv*pv')\eye(n))-1, trace((eye(n) - Dv*P*(Dv^-1) + pv*pv')\eye(n))-1);

A_riem = diag(d)*Xs;
