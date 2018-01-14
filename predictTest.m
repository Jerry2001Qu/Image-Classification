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
## @deftypefn {} {@var{retval} =} predictTest (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Jerry <Jerry@JERRYQU>
## Created: 2018-01-03

function junk = predictTest (sel, Theta1, Theta2, X, y, images)
  load("batches.meta.mat");
  m = size(sel, 2);
  junk = 0;
  
  % Testing labels
  % imshow(squeeze(images(1, :, :, :)));
  % disp(label_names(y(1)));
  
  for i = 1:5
    imshow(squeeze(images(sel(i), :, :, :)));
    % displayData(X(sel(i), :));
    pred = predict(Theta1, Theta2, X(sel(i), :));
    disp(label_names(pred));
    pause;
  end

end
