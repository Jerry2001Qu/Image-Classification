## Copyright (C) 2018 Jerry
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} CostFunction (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Jerry <Jerry@JERRYQU>
## Created: 2018-01-03

function [J grad] = costFunction (nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambda)
  
  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  Theta1 = reshape(nnParams(1:hiddenLayerSize * (inputLayerSize + 1)), ...
                 hiddenLayerSize, (inputLayerSize + 1));

  Theta2 = reshape(nnParams((1 + (hiddenLayerSize * (inputLayerSize + 1))):end), ...
                 numLabels, (hiddenLayerSize + 1));

  % Number of training examples
  m = size(X, 1);
  
  % Setup values to return
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  % ============= FORWARD PROPAGATION ================
  X = [ones(m, 1) X];                   % Add ones for bias nodes
  z_2 = Theta1*X';                      % Calculate input layer z-values
  Hidden1 = sigmoid(z_2);               % Calculate input layer activations
  Hidden1 = [ones(1, m); Hidden1];      % Bias node
  Output = sigmoid(Theta2 * Hidden1);   % Calculate hidden layer activations

  y = eye(numLabels)(y,:);              % Create output matrix with 1's at the respective output node
  y = y';                               % Matrix with dim num_labels(10) x training examples (size(X))

  J = sum(sum((-y .* log(Output)) - ((1 .- y) .* (log(1 .- Output)))))/m;  % Calculate un regularized cost function

  J = J + (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:size(Theta1, 2)) .^ 2)) + sum(sum(Theta2(:, 2:size(Theta2, 2)) .^ 2)));  % Calculate regularization

  % =============== BACK PROPAGATION =================
  z_2 = [ones(1, m); z_2];                                          % Add the bias node for calculating gradient

  for t = 1:m                                                       % Loop over all training examples(m), add up gradients and divide by m
    error_3 = Output(:, t) - y(:, t);                               % Calculate error at output layer
    error_2 = (Theta2' * error_3) .* sigmoidGradient(z_2(:, t));    % Calculate error at hidden layer
    
    error_2 = error_2(2:end);                                       % Take away bias node

    Theta2_grad = Theta2_grad + (error_3 * Hidden1(:, t)');         % Add error for training example t to the Theta2 gradient
    Theta1_grad = Theta1_grad + (error_2 * X(t, :));                % Add error for training example t to the Theta1 gradient
  end

  Theta2_grad = Theta2_grad / m;                                    % Divide by number of training examples (calculate mean)
  Theta1_grad = Theta1_grad / m;

  Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda / m) * Theta1(:, 2:end)); % Add regularization
  Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda / m) * Theta2(:, 2:end));

  grad = [Theta1_grad(:) ; Theta2_grad(:)]; % Unroll Gradients

endfunction
