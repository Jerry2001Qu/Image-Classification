% Setup the parameters
input_layer_size  = 1024;  % 32x32 Input Images of Digits
hidden_layer_size = 81;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that "0" has been mapped to label 10)
                          
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

[X, y, images] = loadData("data_batch_1.mat");
y = y .+ 1;
% dummy = testY(X, y);
m = size(X, 1);

% Normalize
X_mean = sum(X, 2) ./ size(X, 2);
X = (X - X_mean);

% Unroll training data (X)
X = reshape(X, size(X, 1), 1024);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

% displayData(X(sel, :));

% fprintf('Program paused. Press enter to continue.\n');
% pause;

fprintf('\nInitializing Neural Network Parameters ...\n')

% Initialize Theta Params
% Theta1 = randomInitTheta(input_layer_size, hidden_layer_size);
% Theta2 = randomInitTheta(hidden_layer_size, num_labels);

% Import Theta Params
load("param3.mat");

numIters = 5;

[accuracy, Theta1, Theta2] = learningCurve(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, numIters);

save param3.mat Theta1 Theta2;        
                 
fprintf('Program paused. Press enter to continue.\n');
pause;

% fprintf('\nProjecting error... \n')

% plot(1:numIters, accuracy);
% title('Learning Curve for Image Classification (NN)')
% legend('Train')
% xlabel('Number of Iterations')
% ylabel('Accuracy')
% axis([0 numIters 0 100])
% print -djpg Accuracy.jpg;
% pause;

% fprintf('\nVisualizing Neural Network... \n')

% displayData(Theta1(:, 2:end));

% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

junk = predictTest(sel, Theta1, Theta2, X, y, images);