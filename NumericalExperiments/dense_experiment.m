%% Dense Experiment
% This script executes the experiment with dense syntethic Markov chains
% randomly generated.

clear; clc; close all;

addpath('../optimizer/')
addpath('../msrc/mex/')
addpath('../utils/')

nsize       = 120;%[10,50,100,250,500];
nsizes      = length(nsize);
nrepetition = 10;

Kval      = NaN(nsizes,nrepetition);
DeltaNorm = NaN(nsizes,nrepetition);
Kvalopt   = NaN(nsizes,nrepetition);
Kvalkirk  = NaN(nsizes,nrepetition);
PNorm     = NaN(nsizes,nrepetition);

i = 1;
for n = nsize
    fprintf('|----------------------------------------------------------|\n')
    fprintf(['| Problem Size %04d                                    ' ...
        '    |\n'],n)
    fprintf('|----------------------------------------------------------|\n')
    for rep = 1:nrepetition
        %try
            % Generate problem
            Q = rand(n,n);
            D = diag(sum(Q,2));
            Q = D\Q;
            % Compute stationary vector
            [pi,~] = eigs(Q',1,'largestabs');
            pi = pi/sum(pi);
            % Make it reversible
            P = reversible_markov_chain(Q,pi,'barker');
            % Compute the original Kemeny's constant
            Kval(i,rep) = fullkemeny(P,pi);
            % Compute Kirkland's bound
            Kvalkirk(i,rep) = kirkland_bound(pi);
            % Solve the optimization bound
            [Delta,Kvalopt(i,rep)] = optimizekemeny(P, ...
                'stationary',pi,'type','preserving', ...
                'verbose','iter');
            DeltaNorm(i,rep) = norm(Delta,"fro");
            PNorm(i,rep) = norm(P,"fro");
            % We save the result on file for post-processing
            save("dense_experiment.mat","PNorm","DeltaNorm","Kvalopt", ...
                "Kval","Kvalkirk","nsize","nrepetition")
        %catch
        %    fprintf('!! Failed at size %d repetition %d\n',n,rep);
        %    Kval(i,rep) = NaN;
        %    Kvalkirk(i,rep) = NaN;
        %    DeltaNorm(i,rep) = NaN;
        %    PNorm(i,rep) = NaN;
        %end
    end
    i = i+1;
end
% We save the result on file for post-processing
save("dense_experiment.mat","PNorm","DeltaNorm","Kvalopt", ...
    "Kval","Kvalkirk","nsize","nrepetition")

