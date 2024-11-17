%% Test Kemeny Optimization formulations

clear; clc; close all;

addpath('../utils');

%% Generate Markov-Chain
m = 50;
Q = rand(m,m);
D = diag(sum(Q,2));
Q = D\Q;
% Compute stationary vector
[pi,~] = eigs(Q',1,'largestabs');
pi = pi/sum(pi);
% Make the chain reversible
chaintype = input('Select Chain type\n1)Barker\n2)Metropolis-Hastings\n Chaintype = ');
if ~ ismember(chaintype,[1,2])
    chaintype = 1;
end
switch chaintype
    case 1
        fprintf("Selected 'barker'-type chain.\n")
        P = reversible_markov_chain(Q,pi,'barker');
    case 2
        fprintf("Selected 'metropolisì'-type chain.\n")
        P = reversible_markov_chain(Q,pi,'metropolis');
end



%% Inizialize auxiliary quantities
n = size(P,1);
e = ones(n,1);
I = speye(n,n);

%% Compute Original Kemeny's Constant
K = trace( (I - P + e*pi')\I ) - 1;

%% Matrix of equality constraint
Dpi = spdiags(pi,0,n,n);
T   = build_sparse_T(n);
Aeq = [kron(e',I);kron(I,pi');kron(I,Dpi) - kron(Dpi,I)*T];
beq = [zeros(n,1);zeros(n,1);zeros(n^2,1)];
%% Inequality constraints can be expressed as lower-bounds
lb = - P(:);
h = rand(n,1);
h = h./sum(h);
x = zeros(n^2,1);
%% Select implementation of objective and Hessian
sqp = sqrt(pi);
S = (sqp./sqp');
objfun1 = @(Delta) objective(Delta,P,h);
objfun2 = @(Delta) objective_sym(Delta,P,pi,S);
hess = @(Delta,lambda) assemble_H(Delta,lambda,P,pi);
%% Launch solver
selecthessian = input("Hessian and Algorithm:\n1)BFGS\n2)LBFGS\n3)Closed form\n4)SQP\nSelection = ");
if ~ ismember(selecthessian,[1,2,3,4])
    selecthessian = 2;
end
switch selecthessian
    case 1
        options = optimoptions('fmincon','Algorithm','interior-point',...
            'SpecifyObjectiveGradient',true,...
            'HessianApproximation','bfgs',...
            'SubproblemAlgorithm','ldl-factorization',...
            'Display','iter-detailed','Diagnostics','on',...
            'EnableFeasibilityMode',true,'ScaleProblem',true);
    case 2
        options = optimoptions('fmincon','Algorithm','interior-point',...
            'SpecifyObjectiveGradient',true,...
            'HessianApproximation','lbfgs',...
            'SubproblemAlgorithm','factorization',...
            'Display','iter-detailed','Diagnostics','on',...
            'EnableFeasibilityMode',true,'ScaleProblem',true);
    case 3
        options = optimoptions('fmincon','Algorithm','interior-point',...
            'SpecifyObjectiveGradient',true,...
            'HessianFcn',hess,...
            'SubproblemAlgorithm','factorization',...
            'Display','iter-detailed');
    case 4
        options = optimoptions('fmincon','Algorithm','sqp',...
            'SpecifyObjectiveGradient',true,...
            'Display','iter-detailed');
end
[x,~,~,output] = fmincon(objfun2,x,[],[],Aeq,beq,lb,[],[],options);

%% Evaluate the solution

Delta = reshape(x,n,n);
Knew = trace( (I - (P+Delta) + e*h')\I ) - 1;
[pinew,~] = eigs((P+Delta)',1,'largestabs');
pinew = pinew/sum(pinew);

% Compute Kirkland bound
klb = kirkland_bound(pi);

fprintf("Value of Kemeny's constant decreased from %1.3f to %1.3f\n",...
    K,Knew)
fprintf("Kirkland bound: %1.2f\n",klb);
fprintf("Frobenius norm of the perturbation: %1.3e\n",norm(Delta,"fro"));
fprintf("Infinity norm difference of the steady state: %1.2e\n",norm(pi-pinew,"inf"));

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
    Gmat = transpose(INV1*INV1);
    g = Gmat(:) + Delta(:);
end

end

function [f,g] = objective_sym(Delta,P,pi,S)
%%OBJECTIVE Objective function containing Kemeny constant for a dense
% Markov chain.

n = size(P,1);
I = eye(n,n);

Delta = reshape(Delta,n,n);
sqp = sqrt(pi);
Dp = spdiags(sqp,0,n,n);
Dpinv = spdiags(1./sqp,0,n,n);

Q = I - Dp*(P+Delta)*Dpinv + sqp*sqp';
L = chol(Q);

INV1 = L'\(L\I);

f = trace( INV1 ) + 0.5*norm(Delta,"fro")^2;

if nargout > 1
    % Compute the gradient if it is requested
    Gmat = S.*transpose(INV1*INV1);
    g = Gmat(:) + Delta(:);
end

end

function H = assemble_H(Delta,lambda,P,pi)
%ASSEMBLE_H builds the Hessian for the Kemeny's constant function

n = size(P,1);
I = eye(n,n);
Delta = reshape(Delta,n,n);
sqp = sqrt(pi);
Dp = spdiags(sqp,0,n,n);
Dpinv = spdiags(1./sqp,0,n,n);
L = chol(I - Dp*(P+Delta)*Dpinv + sqp*sqp');
INV1 = L'\(L\I);
GMAT = INV1*INV1;

% Define the dimension n
n = size(INV1, 1);
H = zeros(n^2, n^2);  % Initialize the matrix H of size n^2 x n^2

% Nested loops to fill the H matrix
for i=1:n
    for j=1:n
        p = i + (j-1)*n;
        for h=1:n
            for k=1:n
                q = h + (k-1)*n;
                term1 = INV1(j, k); % Compute e_j' * INV1 * e_h
                term2 = GMAT(h, i); % Compute e_k' * GMAT * e_i
                term3 = GMAT(j, k); % Compute e_j' * GMAT * e_h
                term4 = INV1(h, i); % Compute e_k' * INV1 * e_i
                H(p,q) = (sqp(i)/sqp(j))*...
                    (sqp(h)/sqp(k))*(term1 * term2 + term3 * term4);
            end
        end
    end
end

H = H + speye(n^2,n^2);

end