function M = multinomialsymmetricfixedfactory(pv)
% Manifold of n-by-n matrices with positive entries and
% fixed left-hand stochastic eigenvector
%
% function M = multinomialsymmetricfixedfactory(n,pv)
%
% M is a Manopt manifold structure to optimize over the set of n-by-n
% matrices with (strictly) positive entries and such that the elements X
% in M satisfy:
%  X = X^T; Xpv = pv and pv^T X = pv^T,     pv is a n-by-1 vector
%
%
% Points on the manifold and tangent vectors are represented naturally as
% matrices of size n. The Riemannian metric imposed on the manifold is the
% Fisher metric, that is, if X is a point on the manifold and U, V are two
% tangent vectors:
%
%     M.inner(X, U, V) = <U, V>_X = sum(sum(U.*V./X)).
%
% The retraction here provided is only first order.

n = length(pv);
pi = pv.^2;
% maxDSiters = 100 + 2*n;
maxDSiters = 1000 + 2*n;

M.name = @() sprintf(['%dx%d symmetric matrices with positive ' ...
    'entries and fixed right and left eigenvectors'], n, n);

M.dim = @() n*(n-1)/2;

% Fisher metric
M.inner = @iproduct;
    function ip = iproduct(X, eta, zeta)
        ip = sum((eta(:).*zeta(:))./X(:));
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
        X = abs(randn(n, n));
        X = 0.5*(X+X'); 

%        keyboard

        %Retraction con double_stoch_general di manopt
        Xr = diag(pv)*X*diag(pv);
        [XXr, u,v]= my_doubly_stochastic_general(Xr,pi,pi, maxDSiters);
        
        %keyboard

       X = diag(u)*X*diag(v);
       X = 0.5*(X+X'); %probably we don't need it here

       %keyboard
    end

M.randvec = @randomvec;
    function eta = randomvec(X) % A random vector in the tangent space
        % A random vector in the ambient space
        Z = randn(n, n);
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
        b = eta*pv;
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
         Y = X.*exp(t*(eta./X));
 
%         Y = X + t*eta;

        Y = min(Y, 1e50); % For numerical stability
        Y = max(Y, 1e-50); % For numerical stability

    %Retraction con double_stoch_general di manopt
    Yr = diag(pv)*Y*diag(pv);
    [YYr, u, v]= my_doubly_stochastic_general(Yr, pi, pi, maxDSiters);

     Y = diag(u)*Y*diag(v);
     Y = 0.5*(Y + Y');        
     Y = max(Y, eps);      
    end

% Conversion of Euclidean to Riemannian Hessian
M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, eta) %egrad and ehess in the Embedding space, eta in the tangent
        egrad = .5*(egrad + egrad');
        ehess = .5*(ehess + ehess'); %useful if we use AD ?

        % Computing the directional derivative of the Riemannian
        % gradient
        gamma = egrad.*X;
        gammadot = ehess.*X + egrad.*eta;

        bdot = gammadot*pv;
        b = gamma*pv;

        Dv = diag(pv);
        A = Dv*X*Dv + diag(X*Dv*pv);
        alpha = A\b;
       
        S1 = Dv*eta*Dv + diag(eta*Dv*pv);
        alphadot = A \ (bdot - S1*alpha);

        S = (alpha*pv' + pv*alpha');
        deltadot = gammadot - (alphadot*pv' + pv*alphadot').*X - S.*eta; % rgraddot

        % Computing Riemannian gradient
        delta = gamma - S.*X; % rgrad

        % Riemannian Hessian in the ambient space
        nabla = deltadot - 0.5*(delta.*eta)./X;

        % Riemannian Hessian on the tangent space
        rhess = projection(X, nabla);
    end


% Miscellaneous manifold functions
M.hash = @(X) ['z' hashmd5(X(:))];
M.lincomb = @matrixlincomb;
M.zerovec = @(X) zeros(n, n);
M.transp = @(X1, X2, d) projection(X2, d);
M.vec = @(X, U) U(:);
M.mat = @(X, u) reshape(u, n, n);
M.vecmatareisometries = @() false;

end