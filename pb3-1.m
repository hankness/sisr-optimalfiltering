close all
clear all

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

%figure
%plot(pos_vec(1,:),pos_vec(2,:), 'd')
%legend("Stations locations")

%% Mobility tracking constants

v = 90;
eta = 3;
chi = 1.5;  %varsigma
s = 6;
transition_mean = @(x1,x2,l) v - 10*eta*log10(vecnorm([x1;x2] - pos_vec(:,l), 2, 1));

%% SIS Implementation

% Initialization 
X0 = [sqrt(500); sqrt(5); sqrt(5); sqrt(200); sqrt(5); sqrt(5)] .* randn(6,N);
w0 = ones(1,N);
for l = 1:s
    w0 = w0 .* (normpdf(Y(l,1),transition_mean(X0(1,:),X0(4,:),l), chi));
end

% Using log-weights
w0 = log(w0);
maxw = max(w0);
w0 = exp(w0 - maxw);

w0 = w0 / sum(w0);

X = X0;
weights = w0;

list_weights = zeros(m, N); % keeping weights for the histograms
list_weights(1, :) = w0;

driving = unidrnd(5, 1, N); % directions for all particles

taus = zeros(2, m); % position estimates
taus(1, 1) = sum(X(1, :) .* weights) / sum(weights);
taus(2,1) = sum(X(4,:).* weights)/ sum(weights);

effSS = zeros(1,m); % efficient sample size

% Online sampling
d_vec = zeros(1, m);
d_vec(1) = mode(driving);

for k = 2:m
    % Mutation
    W = sigma * randn(2,N);
    driving = drives(cP, driving,N); % new driving command index
    Z = C(:, driving);
    d_vec(k) = mode(driving);
    X = Phi * X + Psiz * Z + Psiw * W; % samples evolution

    % Weighting
    weights = log(weights);
    for l = 1:s
        weights = weights + log(normpdf(Y(l,k),transition_mean(X(1,:),X(4,:),l), chi));
        %weights = weights .* normpdf(Y(l,k), transition_mean(X(1,:),X(4,:),l), chi);
    end

    % Log-weights
    maxw = max(weights);
    weights = exp(weights-maxw);

    weights = weights / sum(weights);

    list_weights(k,:) = weights;

    % Statistical properties
    cvM2 = N * sum((weights ./ sum(weights) - 1/N).^2);
    effSS(k) = N / (1 + cvM2);

    % Filter means
    taus(1,k) = sum(X(1,:).* weights)/ sum(weights);
    taus(2,k) = sum(X(4,:).* weights)/ sum(weights);
end

%% Plotting stations and estimated positions 
figure(1)
plot(taus(1,:), taus(2,:), 'b');
hold on
plot(pos_vec(1,:), pos_vec(2,:), 'rd')
legend('SIS estimates', 'Stations locations')

%% Plotting weight histograms

figure(2)
histogram(log10(list_weights(1,:)), 20, 'Normalization', 'probability');
hold on
histogram(log10(list_weights(10,:)), 20, 'Normalization', 'probability');
hold on
histogram(log10(list_weights(50,:)), 20, 'Normalization', 'probability');
hold on
histogram(log10(list_weights(100,:)), 20, 'Normalization', 'probability');
legend('n = 1','n = 10','n = 50','n = 100','Location','northwest');
xlabel('Importance weights (base 10 logarithm)');
ylabel('Absolute frequency');
title('Importance-weight distribution SIS');


figure(3)
subplot(2, 3, 1)
h1 = histogram(log10(list_weights(1,:)), 20, 'Normalization','probability');
title("n = 1")
xlabel('Importance weights');
ylabel('Absolute frequency');
h1.FaceColor = [0, 0.4470, 0.7410];
h1.EdgeColor = [0, 0.4470, 0.7410];

subplot(2, 3, 2)
h2 = histogram(log10(list_weights(10,:)),20,'Normalization','probability');
title("n = 10")
xlabel('Importance weights');
ylabel('Absolute frequency');
h2.FaceColor = [0.8500, 0.3250, 0.0980];
h2.EdgeColor = [0.8500, 0.3250, 0.0980];

subplot(2, 3, 3)
h3 = histogram(log10(list_weights(50,:)),20,'Normalization','probability');
title("n = 50")
xlabel('Importance weights');
ylabel('Absolute frequency');
h3.FaceColor = [0.9290, 0.6940, 0.1250];
h3.EdgeColor = [0.9290, 0.6940, 0.1250];

subplot(2, 3, 4)
h4 = histogram(log10(list_weights(100,:)),20,'Normalization','probability');
title("n = 100")
xlabel('Importance weights');
ylabel('Absolute frequency');
h4.FaceColor = [0.4940, 0.1840, 0.5560];
h4.EdgeColor = [0.4940, 0.1840, 0.5560];

subplot(2, 3, 5)
h5 = histogram(log10(list_weights(400,:)), 20, 'Normalization','probability');
title("n = 400")
xlabel('Importance weights');
ylabel('Absolute frequency');
h5.FaceColor = [0.4660, 0.6740, 0.1880];
h5.EdgeColor = [0.4660, 0.6740, 0.1880];

subplot(2, 3, 6)
h6 = histogram(log10(list_weights(495,:)), 20, 'Normalization','probability');
title("n = 495")
xlabel('Importance weights');
ylabel('Absolute frequency');
h6.FaceColor = [0.3010, 0.7450, 0.9330];
h6.EdgeColor = [0.3010, 0.7450, 0.9330];
sgtitle("Importance-weight distribution SIS")
%% Plotting efficient sample size
steps = 20;
gap = m / steps; % gap between plotted efficient sample sizes

figure(4)
plot(1:gap:m, effSS(1:gap:end), 'x-')
xlabel('Time step');
legend('Efficient sample size')


%% Functions
function [driving] = drives(cP, driving,N)
    probability = rand(1,N);
    row = cP(driving,:)';

    driving = 1 * (probability < row(1,:)) + 2 * (probability < row(2,:) & probability > row(1,:))... 
    + 3 * (probability < row(3,:) & probability > row(2,:)) ...
    + 4 * (probability < row(4,:) & probability > row(3,:)) ...
    + 5 * (probability < row(5,:) & probability > row(4,:));
end