clear; close all;
m = imread('video 2/image_1887.jpg');
%imshow(m);
img_size = size(m);
X_3_2 = m(591-35:690+35,1716+5:1895-5,:);
imshow(X_3_2);