%% Test Kemeny Optimization formulations

clear; clc; close all;

% Generate Markov-Chain
m = 20;
P = rand(m,m);
D = diag(sum(P,2));
P = D\P;

% Compute stationary vector
[pi,~] = eigs(P',1,'largestabs');
pi = pi/sum(pi);

% Inizialize auxiliary quantities
n = size(P,1);
e = ones(n,1);
I = speye(n,n);

% Compute Original Kemeny's Constant
K = trace( (I - P + e*pi')\I ) - 1;

% Matrix of equality constraint
Aeq = kron(e',I);
beq = zeros(n,1);
% Inequality constraints can be expressed as lower-bounds
lb = - P(:);
options = optimoptions('fmincon','SpecifyObjectiveGradient',true,...
    'HessianApproximation','lbfgs',...
    'Display','iter');
h = rand(n,1);
h = h./sum(h);
x = zeros(n^2,1);
objfun = @(Delta) objective(Delta,P,h);
x = fmincon(objfun,x,[],[],Aeq,beq,lb,[],[],options);

Delta = reshape(x,n,n);
Knew = trace( (I - (P+Delta) + e*h')\I ) - 1;

fprintf("Value of Kemeny's constant decreased from %1.3f to %1.3f\n",...
    K,Knew)
fprintf("Frobenius norm of the perturbation: %1.3e\n",norm(Delta,"fro"));

%% Routines for the optimization
% Implementation of the objective function and gradient (using full
% evaluation of Kemeny's Constant)

function [f,g] = objective(Delta,P,h)
%%OBJECTIVE Objective function containing Kemeny constant for a dense
% Markov chain.

n = size(P,1);
I = eye(n,n);
e = ones(n,1);

Delta = reshape(Delta,n,n);

[L,U] = lu(I - (P+Delta) + e*h');

INV1 = U\(L\I);

f = trace( INV1 ) + 0.5*norm(Delta,"fro")^2;

if nargout > 1
    % Compute the gradient if it is requested
    Gmat = INV1*INV1;
    g = Gmat(:) + Delta(:);
end

end