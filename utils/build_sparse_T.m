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