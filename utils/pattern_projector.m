function [proj] = pattern_projector(S)
%PATTER_PROJECTOR generates the projector onto the pattern for constraint
%variables
k     = find(S);
N    = size(S,1);
proj = sparse(1:1:length(k),k,ones(length(k),1), length(k),N*N );
end