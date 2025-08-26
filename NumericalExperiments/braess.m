%% Depiction of Braess paradox and examples
% Construction of the example with a pendent twin used in the manuscript.
% It generalizes with the number of nodes n.

clear; clc;
addpath("../utils");

n = 10; % Number of nodes
e = ones(n,1);
G = graph(e*e',"omitselfloops");

figure(1)
h = plot(G,"Layout","circle");
x = h.XData;
y = h.YData;

G = G.addnode(2);
G = G.addedge(n,n+1,1);
G = G.addedge(n,n+2,1);
G = G.addedge(n+1,n+2,1);
x = [x,max(x)+1/n,max(x)+1/n];
y = [y,y(end)-0.5,y(end)];

figure(1)
subplot(1,2,1)
plot(G,'k',"XData",x,"YData",y);

P = adjacency(G);
P = diag(sum(P,2))\P;
[h,~] = eigs(P,1,"largestabs");
h = h./sum(h);
K = fullkemeny(P,h);
title(sprintf("$K=%1.2f$",K),"Interpreter","latex")
xticks([])
yticks([])

H = G.rmedge(n+1,n+2);
figure(1)
subplot(1,2,2)
plot(H,'k',"XData",x,"YData",y);
P = adjacency(H);
P = diag(sum(P,2))\P;
[h,~] = eigs(P,1,"largestabs");
h = h./sum(h);
K = fullkemeny(P,h);
title(sprintf("$K=%1.2f$",K),"Interpreter","latex")
xticks([])
yticks([])

set(gcf,'Color','none')
export_fig("braess.pdf")