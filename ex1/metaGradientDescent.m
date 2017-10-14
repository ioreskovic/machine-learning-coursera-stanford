clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];
num_iters = 50;
a = 1;
r = 1 / 3;
alphas =  [0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
alphaColors = "brgkmcy";
theta_init = zeros(3, 1);

figure;
xlabel('Number of iterations');
ylabel('Cost J');
grid on;
hold on;

for i = 1:length(alphas);
  alpha = alphas(i);
  color = alphaColors(i);
  spec = sprintf ('-%s;%f;', color, alpha);
  [theta, J_history] = gradientDescentMulti(X, y, theta_init, alpha, num_iters);
  plot(1:numel(J_history), J_history, spec, 'LineWidth', 2);
end;
