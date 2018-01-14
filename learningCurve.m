
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
## @deftypefn {} {@var{retval} =} learningCurve (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Jerry <Jerry@JERRYQU>
## Created: 2018-01-04

function [accuracy, Theta1, Theta2] = learningCurve (Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, numIters)

accuracy = zeros(numIters, 1);

  for i = 1:numIters
    % Unroll parameters
    nnParams = [Theta1(:) ; Theta2(:)];

    fprintf('\nTraining Neural Network...Iteration \n')
    disp(i);

    %  After you have completed the assignment, change the MaxIter to a larger
    %  value to see how more training helps.
    options = optimset('MaxIter', 10);

    %  You should also try different values of lambda
    lambda = 1;

    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X, y, lambda);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, nnParams, options);

    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    
    % Determine Accuracy
    pred = predict(Theta1, Theta2, X);
    accuracy(i) = mean(double(pred == y)) * 100;
    
    % Save Theta Parameters
    save param3.mat Theta1 Theta2; 
  end
  
end
