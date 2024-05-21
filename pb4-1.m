%% Setup
clear variables
clc

%% Constants
dt = 0.5;
alpha = 0.6;
sigma = 0.5;

P = ones(5,5) / 20 + diag(0.75 * ones(1,5)); % transition probability matrix
cP = cumsum(P,2); % help matrix to evolve the Markov chain 

m = 500; % number of steps
N = 10000; % number of particles 

%% Dynamics matrices
C = [[0; 0], [3.5; 0], [0; 3.5], [0; -3.5], [-3.5; 0]];
z3 = zeros(3,1);

Phi = [1, dt, dt^2 / 2; 0, 1, dt; 0, 0, alpha];
Phi = [Phi, zeros(3); zeros(3), Phi];

Psiz = [dt^2 / 2; dt; 0];
Psiz = [Psiz, z3; z3, Psiz];

Psiw = [dt^2 / 2; dt; 1];
Psiw = [Psiw, z3; z3, Psiw];


%% Data
Y = importdata('RSSI-measurements.mat');
pos_vec = importdata('stations.mat');

%% Mobility tracking constants
v = 90;
eta = 3;
chi = 1.5;   %varsigma
s = 6;
transition_mean = @(x1,x2,l) v - 10*eta*log10(vecnorm([x1;x2] - pos_vec(:,l), 2, 1));
%% SISR Implementation

% Initialization
X0 = [sqrt(500); sqrt(5); sqrt(5); sqrt(200); sqrt(5); sqrt(5)] .* randn(6,N);
w0 = ones(1,N);
for l = 1:s
    w0 = w0 .* (normpdf(Y(l,1),transition_mean(X0(1,:),X0(4,:),l), chi));
end

Xs = zeros(6,N); %keep replicates together to resample
Xs = X0;
weights = w0;
driving = unidrnd(5,1,N); % directions for all particles
taus = zeros(2,m); % position estimates

taus(1,1) = sum(Xs(1,:).* weights)/ sum(weights);
taus(2,1) = sum(Xs(4,:).* weights)/ sum(weights);

effSS = zeros(1,m); % efficient sample size
d_vec = zeros(1, m);
d_vec(1) = mode(driving);

% Online sampling

for k = 2:m
    % Selection 
    ind = randsample(N,N,true,weights); % indices for resampling

    % Mutation
    W = sigma * randn(2,N);
    driving = drives(cP, driving,N); % new driving command index
    Z = C(:,driving);
    d_vec(k) = mode(driving);

    Xs = Phi * Xs(:,ind) + Psiz * Z(:,ind) + Psiw * W; 

    % Weighting
    weights = ones(1,N); % reinitialize weights for the multiplicative property of stations
    for l = 1:s
        weights = weights .* normpdf(Y(l,k),transition_mean(Xs(1,:),Xs(4,:),l), chi);
    end

    % Statistical properties 
    cvN2 = N * sum((weights/sum(weights) - 1/N *ones(1,N)).^2);
    effSS(k) = N / (1 + cvN2);

    % Estimation
    taus(1,k) = sum(Xs(1,:).* weights)/ sum(weights);
    taus(2,k) = sum(Xs(4,:).* weights)/ sum(weights);
end

idx1 = (find(diff(d_vec) == 0));
idx2 = [0 idx1];
idx1 = [idx1 idx2(end)];
avg_stay_t = mean(idx1 - idx2);



% Plotting stations and estimated positions 
figure
plot(taus(1,:), taus(2,:), 'b');
hold on
plot(pos_vec(1,:),pos_vec(2,:), 'd')
legend('SISR estimates', 'Stations locations')
title('Trajectory of target using SISR')



figure
plot(1:m, d_vec, '.')
title('Evolution of driving directions through time')
xlabel('Time steps')
ylabel('Driving directions')
yticklabels({'[0; 0]', '', '[3.5; 0]', '', '[0; 3.5]', '', '[0; -3.5]', '', '[-3.5; 0]'})

%% Efficient sample size

figure
plot(1:m,effSS)
title("Efficient sample size through time")
xlabel('Time steps')
ylabel('Efficient sample size')
%% Functions
function [driving] = drives(cP, driving,N)
    probability = rand(1,N);
    row = cP(driving,:)';

    driving = 1 * (probability < row(1,:)) + 2 * (probability < row(2,:) & probability > row(1,:))... 
    + 3 * (probability < row(3,:) & probability > row(2,:)) ...
    + 4 * (probability < row(4,:) & probability > row(3,:)) ...
    + 5 * (probability < row(5,:) & probability > row(4,:));
end