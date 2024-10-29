function [P,pi] = generate_random_reversible(n,method)
%GENERATE_RANDOM_REVERSIBLE Generates a random irreducible reversable
%Markov chain using either "Barker" or "Metropolis" algorithm.
%   Input:  n size of the chain
%           method "barker" or "metropolis"
%   Output: P reversible and irreducible Markov Chain
%           pi stationary vector of the Chain

pi = zeros(n,1);
while any(abs(pi) < 10*eps)
   P = rand(n,n);
   P = diag(sum(P,2))\P;
   [pi,~] = eigs(P',1,'largestabs');
   pi = pi/sum(pi);
end

P = reversible_markov_chain(P, pi, method);



end

