function K = fullkemeny(P,h)
%FULLKEMENY Evaluates Kemeny by explicitly computing the trace of the
%inverse. This is doable only for small matrices.

n = size(P,1);
e = ones(n,1);
I = eye(n,n);
Z = I - P + e*h';
K = trace(Z\I)-1;

end

