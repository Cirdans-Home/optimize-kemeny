%% Produce Figures for the Dense Experiment Case

clear; clc; close all;

load('dense_experiment.mat');

%% Produce Figures

[X,Y] = meshgrid(1:nrepetition,nsize);
figure(1)
pause(1)
clf
pause(1)
hold on
CO(:,:,1) = zeros(5,10); % red
CO(:,:,2) = ones(5,10).*linspace(0.5,0.6,10); % green
CO(:,:,3) = ones(5,10).*linspace(0,1,10); % blue
surf(X,Y,Kval,CO,'FaceAlpha',0.5);

CO(:,:,1) = zeros(5,10); % red
CO(:,:,2) = ones(5,10).*linspace(0.2,0.3,10); % green
CO(:,:,3) = ones(5,10).*linspace(1,2,10); % blue
surf(X,Y,Kvalopt,CO,'FaceAlpha',0.5);

CO(:,:,1) = ones(5,10); % red
CO(:,:,2) = ones(5,10).*linspace(0.8,1.5,10); % green
CO(:,:,3) = ones(5,10).*linspace(3,4,10); % blue
surf(X,Y,Kvalkirk,CO,'FaceAlpha',0.5);

hold off
set(gca(),'View',[-37.5000 30]);
legend('K(P)','K(P+\Delta)','K(P_{Kirk.})')
set(gca(),'ZScale','log')
xlabel('Repetition')
ylabel('Size of P')
zlabel("Kemeny's Constant")

%% Perturbation

figure(2)
subplot(3,1,1)
heatmap(PNorm(1:3,:))
set(gca,'YData',nsize(1:3));
ylabel('||P||_F');
subplot(3,1,2)
heatmap(DeltaNorm(1:3,:))
set(gca,'YData',nsize(1:3));
ylabel('||\Delta||_F');
subplot(3,1,3)
heatmap(DeltaNorm(1:3,:)./PNorm(1:3,:))
set(gca,'YData',nsize(1:3));
ylabel('||\Delta||_F/||P||_F');
xlabel('Repetition')