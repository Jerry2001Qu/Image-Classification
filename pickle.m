
% Display Image set
load("data_batch_1.mat");
load("batches.meta.mat");
images = reshape(data, size(data, 1), 32, 32, 3);
images = rotdim(images, -1, [3, 2]);
images = sum(images, 4) ./ 3 ./ 255;
disp(squeeze(images(1, :, :, :)))
for i = 1:size(images, 1)
 imshow(squeeze(images(i, :, :, :)));
 disp(label_names(labels(i) + 1));
 pause;
end