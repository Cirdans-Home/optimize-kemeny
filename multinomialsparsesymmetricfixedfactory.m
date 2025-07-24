function M = multinomialsparsesymmetricfixedfactory(pv, S)
% Manifold of n-by-n matrices with non negative entries and
% fixed left-hand stochastic eigenvector
% with an additional (symmetric) sparsity pattern
%
% function M = multinomialsparseymmetricfixedfactory(n,pv)
%
% M is a Manopt manifold structure to optimize over the set of n-by-n
% matrices with nonnegative entries and such that the elements X
% in M satisfy:
%  X = X^T; Xpv = pv and pv^T X = pv^T,     pv is a n-by-1 vector
% Moreover X has a prescribed sparsity pattern, which is SYMMETRIC and has
% the elements on the MAIN DIAGONAL different from zero
%
% Points on the manifold and tangent vectors are represented naturally as
% matrices of size n.
% S is a prescribed sparsity pattern, stored as a matrix of zeros and ones
% Note that: we are assuming that it has always ones on the main diagonal
% and it is symmetric

n = length(pv);
pi = pv.^2;
% maxDSiters = 100 + 2*n;
maxDSiters = 1000 + 2*n;

[I,J] = find(S);
index = sub2ind(size(S), I, J);

M.name = @() sprintf(['%dx%d symmetric matrices with nonnegative ' ...
    'entries and fixed right and left eigenvectors and given sparsity pattern'], n, n);

M.dim = @() 0.5*(nnz(S) - n);

% Fisher metric
M.inner = @iproduct;
    function ip = iproduct(X, eta, zeta)

        %eta,zeta,X have the same sparsity pattern
        nnz_eta = eta(index);
        nnz_zeta = zeta(index);
        nnz_X = X(index);

        ip = sum((nnz_eta(:).*nnz_zeta(:))./nnz_X(:));
    end

M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
 
%M.dist = @(X, Y) error(['multinomialsymmetricfixedfactory.dist not ' ...
%    'implemented yet.']);

% The manifold is not compact as a result of the choice of the metric,
% thus any choice here is arbitrary. This is notably used to pick
% default values of initial and maximal trust-region radius in the
% trustregions solver.
M.typicaldist = @() n;

% Pick a random point on the manifold
M.rand = @random;
    function X = random()
        X = abs(randn(n, n)).*S;
        X = 0.5*(X+X'); 
       % X = X + speye(n,n);

        %Retraction con double_stoch_general di manopt
        Xr = diag(pv)*X*diag(pv);
        [XXr, u,v]= my_doubly_stochastic_general(Xr,pi,pi, maxDSiters);
       X = diag(u)*X*diag(v);
       X = 0.5*(X+X'); %probably we don't need it here
    end

M.randvec = @randomvec;
    function eta = randomvec(X) % A random vector in the tangent space
        % A random vector in the ambient space
        Z = randn(n, n).*S; % REMARK: we assume the sparsity pattern contains the main diagonal
        Z = 0.5*(Z+Z');
        % Projection of the vector onto the tangent space
        b = Z*pv;
        Dv = diag(pv);
        A = Dv*X*Dv + diag(X*Dv*pv);
        
        alpha = A\b;
        eta = Z - (alpha*pv' + pv*alpha').*X;

        % Normalizing the vector
        nrm = M.norm(X, eta);
        eta = eta / nrm;
    end

M.proj = @projection;
    function etaproj = projection(X, eta) % Projection of the vector eta onto the tangent space

        eta = 0.5*(eta + eta');
        eta = eta.*S;
        b = (eta)*pv;

        Dv = diag(pv);
        A = Dv*X*Dv + diag(X*Dv*pv);
        alpha = A\b;
        etaproj = eta - (alpha*pv' + pv*alpha').*X; 
   end

M.tangent = M.proj;
M.tangent2ambient = @(X, eta) eta;

% Conversion of Euclidean to Riemannian gradient
M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad) % projection of the euclidean gradient
        
        egrad = egrad.*S;
        egrad = 0.5*(egrad + egrad'); %projection onto the ambient space
      
        mu = (X.*egrad);

        b = mu*pv;
        Dv = diag(pv);
        A = Dv*X*Dv + diag(X*Dv*pv);
        alpha = A\b;
        rgrad = mu - (alpha*pv' + pv*alpha').*X;


    end

% First-order retraction
M.retr = @retraction;
    function Y = retraction(X, eta, t)
       if nargin < 3
           t = 1.0;
       end
      
       XX = X;
       XX(XX==0) = 1;

       Y = X.*exp(t*(eta./XX));
       Y = Y.*S;

    %Y = X + t*eta;
     Y = max(Y, 1e-30); % For numerical stability;
     Y = Y.*S;

    %Retraction con double_stoch_general di manopt
    Yr = diag(pv)*Y*diag(pv);
    [YYr, u, v]= my_doubly_stochastic_general(Yr, pi, pi, maxDSiters);

     Y = diag(u)*Y*diag(v);
     Y = Y.*S;
     Y = 0.5*(Y + Y');
    end

 M.ehess2rhess = @(X, egrad, ehess, eta) error(['Hessian not ' ...
   'implemented yet.']);

% Miscellaneous manifold functions
M.hash = @(X) ['z' hashmd5(X(:))];
M.lincomb = @matrixlincomb;
M.zerovec = @(X) zeros(n, n);
M.transp = @(X1, X2, d) projection(X2, d);
M.vec = @(X, U) U(:);
M.mat = @(X, u) reshape(u, n, n);
M.vecmatareisometries = @() false;

end