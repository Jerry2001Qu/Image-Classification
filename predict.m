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
## @deftypefn {} {@var{retval} =} predict (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Jerry <Jerry@JERRYQU>
## Created: 2018-01-03

function p = predict (Theta1, Theta2, X)

  m = size(X, 1);

  h1 = sigmoid([ones(m, 1) X] * Theta1');
  h2 = sigmoid([ones(m, 1) h1] * Theta2');
  [dummy, p] = max(h2, [], 2);

endfunction
