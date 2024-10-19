function Q = reversible_markov_chain(P, pi, method)
%REVERSIBLE_MARKOV_CHAIN takes as input an ergodic stochastic matrix P and
%its stationary vector pi and produces a stochastic matrix Q of a
%reversible Markov chain with stationary vector pi.
% Input:
% P      - Transition matrix (n x n)
% pi     - Stationary distribution (1 x n), must be a row vector
% method - 'barker' or 'metropolis', to specify which algorithm to use
% Output:
% Q      - Reversible Markov matrix (n x n)

% Check dimensions
n = size(P, 1);
if length(pi) ~= n
    error('Stationary distribution vector size must match matrix size.');
end
if norm(pi'*P - pi') > 50*eps || abs(sum(pi)-1) > 50*eps
    error('The vector pi is not stationary a vector');
end

% Initialize reversible matrix Q
Q = zeros(n, n);

% Loop over all entries of the transition matrix
for i = 1:n
    for j = 1:n
        if i ~= j
            % Get the proposal probabilities P(i, j) and P(j, i)
            P_ij = P(i, j);
            P_ji = P(j, i);
            
            % Apply the chosen method
            switch lower(method)
                case 'barker'
                    % Barker acceptance probability
                    alpha_ij = pi(j) * P_ji / (pi(i) * P_ij + pi(j) * P_ji);
                case 'metropolis'
                    % Metropolis-Hastings acceptance probability
                    alpha_ij = min(1, (pi(j) * P_ji) / (pi(i) * P_ij));
                otherwise
                    error('Unknown method. Use "barker" or "metropolis".');
            end
            
            % Adjust Q(i, j) and Q(j, i) to ensure reversibility
            Q(i, j) = P_ij * alpha_ij;
            Q(j, i) = P_ji * (pi(i) * Q(i, j)) / (pi(j) * P_ji); % Detailed balance condition
        end
    end
end

% Set diagonal elements of Q such that rows sum to 1
for i = 1:n
    Q(i, i) = 1 - sum(Q(i, :));
end
end