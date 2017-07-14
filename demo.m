clear,close all,clc;

load data.mat;
ENL=3;
tic;
img_filtered = SAR_POTDF(img,'ENL',ENL);
toc;
figure,imshow([img,img_filtered],[0 0.02]);