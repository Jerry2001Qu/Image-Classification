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
## @deftypefn {} {@var{retval} =} loadData (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Jerry <Jerry@JERRYQU>
## Created: 2018-01-04

function [X, y, images]  = loadData (file)
  load(file);
  load("batches.meta.mat");
  images = reshape(data, size(data, 1), 32, 32, 3);
  y = labels;
  
  load("data_batch_2.mat");
  images = [images; reshape(data, size(data, 1), 32, 32, 3)];
  y = [y; labels];
  
  load("data_batch_3.mat");
  images = [images; reshape(data, size(data, 1), 32, 32, 3)];
  y = [y; labels];
  
  load("data_batch_4.mat");
  images = [images; reshape(data, size(data, 1), 32, 32, 3)];
  y = [y; labels];
  
  images = rotdim(images, -1, [3, 2]);
  X = sum(images, 4) ./ 3 ./ 255;
  
  % disp(squeeze(images(1, :, :, :)))
  % for i = 1:size(images, 1)
  %  imshow(squeeze(images(i, :, :, :)));
  %  disp(label_names(labels(i) + 1));
  %  pause;
  % end
end
