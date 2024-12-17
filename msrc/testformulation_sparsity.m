%% Test Kemeny Optimization formulations

clear; clc; close all;

addpath('../utils');
addpath('mex');

% Generate Random Reversible (sparse) Chain
n = 60;
p = 0.05;
A = sprand(n,n,p);
A = spones(A+A');
D = spdiags(sum(A,2),0,n,n);
P = D\A;

[pi,~] = eigs(P',1,'largestabs');
pi = pi/sum(pi);

%% Inizialize auxiliary quantities
n = size(P,1);
e = ones(n,1);
I = speye(n,n);

%% Compute Original Kemeny's Constant
K = trace( (I - P + e*pi')\I ) - 1;

figure(1)
subplot(1,3,1)
spy(P)
fprintf('Initial Kemeny Constant: %1.2f\n',K);

%% Optimize
[Delta,Kopt] = optimizekemeny(P,'type','SparsePreserving',...
    'stationary',pi,...
    'pattern',A,...
    'verbose','iter');
%% Visualize
fprintf('Norm of Delta is: %1.2e\n',norm(Delta,"fro"));
fprintf('Optimized Kemeny Constant is: %1.2f\n',Kopt);
fprintf('Check Constraints:\n')
fprintf("\tnorm(pi'*(P+Delta)-pi') = %e\n",norm(pi'*(P+Delta)-pi'));
Phat = P+Delta;
fprintf("\tnorm( Phat(Phat < 0) ) = %e\n",norm( Phat(Phat < 0) ));
fprintf("\tnorm(diag(pi)*Phat - Phat'*diag(pi)) = %e\n",norm(diag(pi)*Phat - Phat'*diag(pi)));
figure(1)
subplot(1,3,2)
hold on
spy(Delta > 0,'+');
spy(Delta < 0,'r_');
hold off
xlabel(sprintf("Positive %d Negative %d",nnz(Delta > 0),nnz(Delta < 0)))
subplot(1,3,3)
[ix, iy] = ind2sub([n, n], find(Delta));
[~, ~, vector] = find(Delta);
vector_norm = (vector - min(vector)) / (max(vector) - min(vector));
colormap_choice = parula; 
n_colors = size(colormap_choice, 1);
color_indices = round(vector_norm * (n_colors - 1)) + 1;
rgb_array = colormap_choice(color_indices, :);
scatter(ix, iy, [], rgb_array, 'Marker', '.', 'SizeData', 200);
colormap(colormap_choice); % Set the colormap for the figure
c = colorbar; % Create colorbar
clim([min(vector), max(vector)]); % Scale the colorbar to the range of the original vector
set(gca, 'YDir','reverse')
xlabel(sprintf('$\\| \\Delta \\|_F = %1.2e$',norm(Delta,"fro")),...
    'Interpreter','latex')