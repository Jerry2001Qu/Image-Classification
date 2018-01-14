load("batches.meta.mat");
load("param2.mat");

%% param2.mat - OVERFITTING DATA_BATCH_1
%% param3.mat - GENERALIZED (60% on Familiar, 40% on Unfamiliar

[X, y, images] = loadDataClassify("data_batch_1.mat");
m = size(X);
y = y .+ 1;

% Normalize
X_mean = sum(X, 2) ./ size(X, 2);
X = (X - X_mean);

% Unroll training data (X)
X = reshape(X, size(X, 1), 1024);

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

idx = randperm(m);
xTemp = X;
X = squeeze(X(:, :, :));

X(idx, :) = xTemp(:, :);
images(idx, :, :, :) = images(:, :, :, :);

for i = 1:m
  imshow(squeeze(images(i, :, :, :)));
  pred = predict(Theta1, Theta2, X(i, :));
  disp(label_names(pred));
  pause;
end