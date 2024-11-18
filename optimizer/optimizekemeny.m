function [Delta,varargout] = optimizekemeny(P,varargin)
%OPTIMIZEKEMENY Find peturbation Delta of minimal norm delivering a new
%Markov chain P+Delta with smaller Kemeny constant.
% Inputs:
%   P (Required)
%       Type: Numeric matrix (must be numeric)
%       Description: A required input parameter, the row-stochastic matrix
%       representing the Markov chain
%
%   type (Optional)
%       Type: String
%       Default: 'Unstructured'
%       Valid Options: 'Unstructured', 'Preserving', 'Sparse', 'SparsePreserving'
%       Description: Specifies the type of operation or structure to be 
%       applied in the optimization. It can take one of the valid options listed
%       above, each corresponding to different processing types:
%           - 'Unstructured' assumes that P is dense and minimize Kemeny's
%           constant. 
%           - 'Preserving' assumes that P is sparse and that we want to
%           preserve the stationary vector, that has to be passed to the
%           routine (see next)
%           - 'Sparse' assumes that P is sparse and minimize Kemeny's
%           constant with an assigned sparsity pattern (see next). 
%           - 'SparsePreserving' assumes that P is sparse and that we want
%           to preserve the stationary vector, that has to be passed to the
%           routine (see next). It minimizes Kemeny's constant with an 
%           assigned sparsity pattern (see next). 
%
%   stationary (Optional)
%       Type: Numeric vector (size [size(P,1), 1])
%       Default: NaN(size(P,1),1)
%       Description: A numeric vector, which by default is a vector of NaN 
%       values with the same number of rows as P. It represent the 
%       stationary vector of the Markov chain (is used only with types
%       'Preserving' and 'SparsePreserving'.
%
%   pattern (Optional)
%       Type: Sparse matrix
%       Default: spones(P) (sparse matrix with ones in the same non-zero 
%       positions as P)
%       Description: A sparse matrix pattern based on the input matrix P. By 
%       default, it is derived from P using spones(P), which creates a sparse
%       matrix with ones in place of non-zero elements of P.
%
%   h (Optional)
%       Type: Numeric vector (size [size(P,1), 1])
%       Default: rand(size(P,1),1)
%       Description: A numeric vector that represents the rank-1 one
%       correction to be used in the evaluation of Kemeny's constant
%
%   verbose (Optional)
%       Type: String
%       Default: 'final'
%       Valid Options: 'final', 'iter'
%       Description: Controls the level of verbosity or output during function
%       execution. The options allow for final output only ('final') or iterative 
%       output ('iter') depending on the level of detail required.
%
% Validations:
%   - P must be numeric.
%   - type must be one of 'Unstructured', 'Preserving', 'Sparse', or 
%     'SparsePreserving'.
%   - stationary must be numeric and of size [size(P,1), 1].
%   - pattern must be a sparse matrix.
%   - h must be numeric and of size [size(P,1), 1].
%   - verbose must be either 'final' or 'iter'.


%% Create input parser
p = inputParser;
validType = {'Unstructured','Preserving','Sparse','SparsePreserving'};
defaultType = 'Unstructured';
checkType = @(x) any(validatestring(x,validType));
validVerbose = {'final','iter'};
checkVerbose = @(x) any(validatestring(x,validVerbose));

addRequired(p,'P',@isnumeric);
addOptional(p,'type',defaultType,checkType);
addOptional(p,'stationary',NaN(size(P,1),1),@isnumeric)
addOptional(p,'pattern',spones(P),@issparse);
addOptional(p,'h',rand(size(P,1),1),@isnumeric);
addOptional(p,'verbose','final',checkVerbose);

parse(p,P,varargin{:});

%% Chek P
if size(P,1) ~= size(P,2)
   error('OptimizeKemeny needs a square matrix');
end
if norm(sum(P,2) - 1,"inf") > 1e-14
    error('OptimizeKemeny needs a row-stochastic matrix');
end

%% Check outputs
if nargout > 2
    error('OptimizeKemeny: more than two output requested');
end

%% Call the appropriate solver
switch upper(p.Results.type)
    case 'UNSTRUCTURED'
        % Solver for dense matrix P without assumptions
        if abs(sum(p.Results.h) - 1) > 10*eps
            h = p.Results.h./sum(p.Results.h);
        else
            h = p.Results.h;
        end
        Delta = unstructured_solver(P,h,p.Results.verbose);
    case 'PRESERVING'
        % Solver for dense matrix P enforcing preservation of the
        % stationary vector
        if size(P,1) ~= length(p.Results.stationary)
            error('OptimizeKemeny: stationary vector has wrong size %d',length(p.Results.stationary))
        end
        if any(isnan(p.Results.stationary))
            error('OptimizeKemeny: did you pass the stationary vector? there are NaN in here');
        end
        if abs(sum(p.Results.stationary) - 1) > 10*eps
            pi = p.Results.stationary./sum(p.Results.stationary);
        else
            pi = p.Results.stationary;
        end
        Delta = preserving_solver(P,pi,p.Results.verbose);
    case 'SPARSE'
        % Solver for obtaining a sparse perturbation without preservation
        % of the stationary vector
        if abs(sum(p.Results.h) - 1) > 10*eps
            h = p.Results.h./sum(p.Results.h);
        else
            h = p.Results.h;
        end
        if size(p.Results.pattern,1) ~= size(p.Results.pattern,1) || ...
                any(size(p.Results.pattern) ~= size(P))
            error("OptimizeKemeny: the pattern is either rectangular or has wrong dimension");
        end
        Delta = sparse_solver(P,p.Results.pattern,h,p.Results.verbose);
    case 'SPARSEPRESERVING'
        
end
    
if nargout == 2
    % The new Kemeny constant value has been requested:
    n = size(P,1);
    I = eye(n,n);
    e = ones(n,1);
    if abs(sum(p.Results.h)-1) > 1e-14
        h = p.Results.h./sum(p.Results.h);
    else
        h = p.Results.h;
    end
    varargout{1} = trace( (I - (P+Delta) + e*h')\I ) - 1;
end


end

%% Solvers

function Delta = unstructured_solver(P,h,verbose)
%%UNSTRUCTURED_SOLVER Solve the optimization problem assuming P a dense
%matrix and that no requirement on is structure is made.

% Inizialize auxiliary quantities
n = size(P,1);
e = ones(n,1);
I = speye(n,n);

Aeq = kron(e',I);
beq = zeros(n,1);
% Inequality constraints can be expressed as lower-bounds
lb = - P(:);
options = optimoptions('fmincon','Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,...
    'HessianApproximation','lbfgs',...
    'Display',verbose);
x = zeros(n^2,1);
objfun = @(Delta) objective(Delta,P,h);
x = fmincon(objfun,x,[],[],Aeq,beq,lb,[],[],options);
Delta = reshape(x,n,n);
end

function Delta = preserving_solver(P,pi,verbose)
%%UNSTRUCTURED_SOLVER Solve the optimization problem assuming P a dense
%matrix and that the original stationary vector has to be preserved
% Inizialize auxiliary quantities
n = size(P,1);
if n > 50
    solver = 'cg';
else
    solver = 'ldl-factorization';
end
e = ones(n,1);
I = speye(n,n);
% Matrix of equality constraint
Dpi = spdiags(pi,0,n,n);
T   = build_sparse_T(n);
Aeq = [kron(e',I);kron(I,pi');kron(I,Dpi) - kron(Dpi,I)*T];
beq = [zeros(n,1);zeros(n,1);zeros(n^2,1)];
% Inequality constraints can be expressed as lower-bounds
lb = - P(:);
if exist('assemble_hessian.mexa64','file') == 3
    fprintf("Found Hessian mex file implementation.\n")
    hess = @(Delta,lambda) assemble_H_mex(Delta,lambda,P,pi);
else
    fprintf("Using purely MATLAB Hessian.\n")
    hess = @(Delta,lambda) assemble_H(Delta,lambda,P,pi);
end
options = optimoptions('fmincon','Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,...
    'HessianFcn',hess,...
    'SubproblemAlgorithm',solver,...
    'Display',verbose);
x = zeros(n^2,1);
sqp = sqrt(pi);
S = (sqp./sqp');
objfun = @(Delta) objective_sym(Delta,P,pi,S);
x = fmincon(objfun,x,[],[],Aeq,beq,lb,[],[],options);
Delta = reshape(x,n,n);
end

function Delta = sparse_solver(P,S,h,verbose)
%%UNSTRUCTURED_SOLVER Solve the optimization problem assuming P a sparse
%matrix and that there is a restricted pattern on which we wish to solve

% Inizialize auxiliary quantities
n = size(P,1);
m = nnz(S);
e = ones(n,1);
I = speye(n,n);
% Build pattern index
[ival,jval] = find(S);
% Matrix of equality constraint
[proj] = pattern_projector(S);
Aeq = kron(e',I)*proj'; 
beq = zeros(n,1);
% Inequality constraints can be expressed as lower-bounds
lb = - proj*P(:);
options = optimoptions('fmincon','Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,...
    'HessianApproximation','lbfgs',...
    'Display',verbose);
x = zeros(m,1);
objfun = @(Delta) objective_sparse(Delta,P,h,ival,jval);
x = fmincon(objfun,x,[],[],Aeq,beq,lb,[],[],options);
Delta = sparse(ival,jval,x,n,n);
end

%% Objective functions
% These routines implement the different evaluation strategies for the
% objective function

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

function [f,g] = objective_sparse(Delta,P,h,ival,jval)
%%OBJECTIVE Objective function containing Kemeny constant for a dense
% Markov chain.
n = size(P,1);
I = eye(n,n);
e = ones(n,1);

Deltamat = sparse(ival,jval,Delta,n,n);
[L,U] = lu(I - (P+Deltamat) + e*h');
INV1 = U\(L\I);
f = trace( INV1 ) + 0.5*norm(Delta,"fro")^2;
if nargout > 1
    % Compute the gradient if it is requested
    Gmat = INV1*INV1;
    index = sub2ind([n,n],ival,jval);
    g = Gmat(index) + Delta;
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

%% Hessian Computation
% These routines implement the computation of the Hessian matrix for the
% optimization problem
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

function H = assemble_H_mex(Delta,lambda,P,pi)
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

H = assemble_hessian(INV1,GMAT,n);

end


%% Utilities
% These function are used to build the matrices of the constraints for the
% optimization problems.
function [proj] = pattern_projector(S)
%PATTER_PROJECTOR generates the projector onto the pattern for constraint
%variables
k     = find(S);
N    = size(S,1);
proj = sparse(1:1:length(k),k,ones(length(k),1), length(k),N*N );
end

function T = build_sparse_T(n)
%%BUILD_SPARSE_T This function builds a sparse matrix T such that
%T*v = w where v = reshape(P', n^2, 1) and w = P(:)

% Initialize arrays to store row, column indices and values
row_idx = zeros(n^2, 1);  % Row indices for sparse matrix
col_idx = zeros(n^2, 1);  % Column indices for sparse matrix
values = ones(n^2, 1);    % Values (all ones)
% Loop over each element of the matrix P
for i = 1:n
    for j = 1:n
        % Linear index for element (i, j) in matrix P(:)
        row = (j - 1) * n + i;
        % Linear index for element (i, j) in matrix reshape(P',n^2,1)
        col = (i - 1) * n + j;      
        % Store indices in sparse matrix
        row_idx(row) = row;
        col_idx(row) = col;
    end
end
% Build the sparse matrix T
T = sparse(row_idx, col_idx, values, n^2, n^2);
end