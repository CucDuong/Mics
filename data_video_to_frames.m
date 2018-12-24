clear; close all;clc;
v = VideoReader('How many living things are in a drop of  dirty water_.mp4');
frameCount=0;
while (v.hasFrame())
    frameCount = frameCount+1;
    m = v.readFrame();
    close;
    imshow(m);
    reply = input('Do you want to save this image (Y/N) ?','s');
    if (~isempty(reply))
        img_name = "image_"+num2str(frameCount)+".jpg";
        imwrite (m,char(img_name) ,'JPEG');
    end
end