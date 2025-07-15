%Comparison between Riemannian and Euclidean
clear all; close all; clc
rng(15)
n  = 10;

run Script_Riemm_Kemeny.m

%Additional checks on the Reversibility - RIEMANNIAN CASE
norm(diag(pi)*Xs - Xs'*diag(pi))
norm(Xs*ones(n,1) - ones(n,1))
norm(pi'*Xs - pi')

fprintf("||Xs - Bhat|| = %e\n", norm(Xs - Bhat,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*Xs*(Dv^-1) + pv*pv')\eye(n)), trace((eye(n) - Dv*Bhat*(Dv^-1) + pv*pv')\eye(n)));

[Delta,varargout] = optimizekemeny(Bhat, 'SparsePreserving', pi);

%Additional checks on the Reversibility - EUCLIDEAN CASE
norm(diag(pi)*(Bhat+Delta) - (Bhat+Delta)'*diag(pi))
norm((Bhat+Delta)*ones(n,1) - ones(n,1))
norm(pi'*(Bhat+Delta) - pi')

fprintf("||(Bhat+Delta) - Bhat|| = %e\n", norm((Bhat+Delta) - Bhat,"fro"));
fprintf("New Kemeny vs Old Kemeny = %e %e\n", trace((eye(n) - Dv*(Bhat+Delta)*(Dv^-1) + pv*pv')\eye(n)), trace((eye(n) - Dv*Bhat*(Dv^-1) + pv*pv')\eye(n)));
