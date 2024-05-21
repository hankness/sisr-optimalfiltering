%% Constants
dt = 0.5;
alpha = 0.6;
sigma = 0.5;
P = ones(5,5) / 20 + diag(0.75 * ones(1,5)); % transition probability matrix
cP = cumsum(P,2); % help matrix to evolve the Markov chain 
m = 1000; % number of steps
directions = [0 0 0 0 0];

%% Dynamics matrices
C = [[0; 0], [3.5; 0], [0; 3.5], [0; -3.5], [-3.5; 0]];
z3 = zeros(3,1);

Phi = [1, dt, dt^2 / 2; 0, 1, dt; 0, 0, alpha];
Phi = [Phi, zeros(3); zeros(3), Phi];

Psiz = [dt^2 / 2; dt; 0];
Psiz = [Psiz, z3; z3, Psiz];

Psiw = [dt^2 / 2; dt; 1];
Psiw = [Psiw, z3; z3, Psiw];

%% Initialization
X0 = [sqrt(500); sqrt(5); sqrt(5); sqrt(200); sqrt(5); sqrt(5)] .* randn(6,1);
driving = unidrnd(5);
Z = C(:, driving);

%% Evolution
X = X0;
Xlist = ones(6,m+1);
Xlist(:,1) = X0;

for i=1:m
    W = sigma * randn(2,1);
    driving = drive(cP, driving); % new driving command index
    Z = C(:,driving);
    X = Phi * X + Psiz * Z + Psiw * W;
    Xlist(:,i+1) = X;
end

%% Plot
figure
plot(Xlist(1,1), Xlist(4,1), 'gd');
hold on

plot(Xlist(1,:), Xlist(4,:), 'b--');
hold on

title("Simulated trajectory for m = " + m + " time steps")
xlabel("X1 coordinate")
ylabel("X2 coordinate")
legend("X0", "Trajectory")

%% Functions
function [driving] = drive(cP, driving)
    probability = rand;
    row = cP(driving,:);

    if probability < row(1)
       driving = 1;
    elseif probability < row(2)
       driving = 2;
    elseif probability < row(3)
       driving = 3;
    elseif probability < row(4)
       driving = 4;
    else
       driving = 5;
    end
end