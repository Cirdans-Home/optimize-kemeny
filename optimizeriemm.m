function [Xf, xcost] = optimizeriemm(A,pi,S)
%optimize Kemeny's constant via Riemannian optimization 
%over the set of reversible matrices with prescribed pattern

%A = initial matrix
%pi = target stationary vector
%S = given sparsity pattern

n = length(pi);

%Qualche check su S, P e pi [da Inserire] 
pv = pi.^(1/2);

tol = logspace(-3, -12, 4);

M = multinomialsparsesymmetricfixedfactory(pv,S);
Bs = M.rand();

Sp = S;

for i = 1:length(tol)
    
    clear M;
    %Initialize the manifold and an initial point
    M = multinomialsparsesymmetricfixedfactory(pv,Sp);
    
    %Initialization of the problem
    problem.M = M;
    problem.cost = @(X) trace((eye(n) - X + pv*pv')\eye(n)) + 0.5*norm(diag(pv.^(-1))*X*diag(pv) - A,'fro')^2;
    problem.egrad = @(X) ((eye(n) - X + pv*pv')^2\eye(n))' + (diag(pi.^(-1))*X*diag(pi) - diag(pv.^(-1))*A*diag(pv));
     
    clear options;
    options.tolgradnorm = tol(i);
    options.verbosity = 0;
%   options.strategy = 'alternate';
%   options.ls_nmsteps = 5;
     
    [X1, xcost, info, options] = conjugategradient(problem, Bs, options);
    %[X1, xcost, info, options] = barzilaiborwein(problem, Bs, options);

    Sp = X1;
    Sp(abs(Sp)<1e-20) = 0;
    Sp = full(spones(Sp+eye(n)));

    Bs = X1;
end

Xf = diag(pv.^(-1))*X1*diag(pv);
