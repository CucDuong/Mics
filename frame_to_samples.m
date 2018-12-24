clear;
m = imread('video 2/image_1909.jpg');
img_size = size(m);
width = 120;
height = 120;
x_count = floor(img_size(1)/height);
y_count = floor(img_size(2)/width);
X_120=zeros(height,width,3,x_count*y_count);
Y_120=zeros(x_count*y_count,1);
for i=1:x_count
    for j=1:y_count
        sample = m((i-1)*height+1 : i*height,(j-1)*width+1 : j*width,:);
        close all;
        imshow(sample);
        X_120(:,:,:,(i-1)*y_count+j) = sample;
        reply = input('What is the class of this sample?');
        Y_120((i-1)*y_count+j)=reply;
    end
end
save X_120;
save Y_120;
 