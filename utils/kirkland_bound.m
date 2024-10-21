function klb = kirkland_bound(pi)
%KIRKLAND_BOUND Given the stationary vector pi of an irreeducible Markov
%chain computes the lower-bound on Kemeny's constant for an irreducible
%chain with pi as stationary vector.


% Check inputs
if any(pi <= 0)
    error("pi has some entries <= 0");
end
if abs(sum(pi)-1) > 10*eps
   warning("Given unnormalized pi, I'm normalizing it"); 
   pi = pi./sum(pi);
end

% build the sorted version of pi
spi = sort(pi,'ascend');
% compute the bound
n = length(pi);
j = (1:n).';
klb = sum((j-1).*spi);

end

