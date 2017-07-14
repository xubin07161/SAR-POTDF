function [IOut2,IOut1,Dict] = SAR_POTDF(img,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  POTDF is an algorithm for SAR image despeckling. 
%  This algorithm reproduces the results from the article:
%  [1] B. Xu et al. 'Patch Ordering based SAR Image Despeckling via 
%      Transform-Domain Filtering'
%  Please refer to this paper for a more detailed description of the algorithm.
%
%  BASIC USAGE EXAMPLES:
%
%     1) Using the default parameters
% 
%      img_filtered = SAR_POTDF(img,'ENL',ENL)
% 
%  INPUT ARGUMENTS (OPTIONAL):
%
%     1) img : The input SAR image should be intensity image.
%
%     2) ENL : The equivalent number of looks. The ENL can be obtained by 
%              by supervised or unsupervised estimation. For a homogeneous 
%              region, the ENL can be calculated by ENL=(mean)^2/var
%
%  OUTPUTS:
%     1) img_filtered  : The filtered intensity image                                             
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (c) 2014 Bin Xu.
% All rights reserved.
% This work should only be used for nonprofit purposes.
%
% AUTHORS:
%     Bin Xu, email: xubin07161@gmail.com
%     Bin Zuo, email: zuob2009@126.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

par.PtSz1=8;                % PtSz stands for Patch Size;i.e. sqrt(n_1) in the paper
par.PtSrSz1=9;              % PtSrSz stands for Patch Search Size;i.e. (C+1)/2 in the paper
par.PtSz2=6;                % sqrt(n_2) in the paper
par.PtSrSz2=9;              % PtSrSz stands for Patch Search Size;i.e. (C+1)/2 in the paper
par.BcSz=3;                 % BcSz stands for Boxcar Size;
par.SlidDt=2;               % SlidDt stands for Sliding Distance;i.e. SL_1^(p) in the paper
par.SlidDt2=1;              % SL_2^(p) in the paper
par.K=512;                  % k in the paper,dictionary size
par.learnDic=1;             % learned dictionary
par.waveType='haar';        % use 2-D haar wavelet with 4-level decomposition

% consider the odd case
[IRow,ICol]=size(img);
if mod(IRow,2)==1
    img(IRow+1,:)=img(IRow,:);
end
if mod(ICol,2)==1
    img(:,ICol+1)=img(:,ICol);
end

% the following parameters can be defined 
for argI = 1:2:length(varargin)
    if (strcmp(varargin{argI}, 'ENL'))
        par.ENL = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'SD'))
        par.SlidDt = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'SD2'))
        par.SlidDt2 = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'sigma'))
        par.sigma = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'wave'))
        par.waveType = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'K'))
        par.K = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'n1'))
        par.PtSz1 = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'n2'))
        par.PtSz2 = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'BcSz'))
        par.BcSz = varargin{argI+1};
    end
end
%% Step 1
% see Section 2(A) in the paper
n=par.PtSz1;                % sqrt(n) in Step 1
par.sigma=sqrt(psi(1,par.ENL)); % variance, used in Step 1 and Step 2 
par.lamda=0.95*par.sigma; % threshold, used in Step 2, see Section 2(D) in the paper

img(isnan(img))=0; % consider the NAN case
img=abs(img); % consider the nonnegative case
img(img==0)=10^(-12); % consider the zero case

% Logarithmic transformation with bias correction
imgLog=log(img)-(psi(0,par.ENL)-log(par.ENL));
% Boxcar filter
imgBoxcar=imfilter(img,ones(par.BcSz,par.BcSz)/(par.BcSz)^2);

imgSz=size(img);           %原始图像的大小
subImgSz=floor((imgSz-[n n])/par.SlidDt+[1 1]);   %每个子图的大小

% Extract patches
[imgCol,~] = f_im2col(log(imgBoxcar),[n n],par.SlidDt);
% Patch ordering
blkOrder=c_PatchSort( imgCol,n,subImgSz );

[imgCol,idx] = f_im2col(imgLog,[n n],par.SlidDt);
[blkLen,numOfBlks]=size(imgCol);       %块的数量
imgCol=imgCol(:,blkOrder); 

% subtract the mean
vecOfMeans = mean(imgCol);
imgCol = imgCol-ones(blkLen,1)*vecOfMeans;

% Create an initial dictionary from the DCT frame
Pn=ceil(sqrt(par.K));
DicOri=Generate_DCT_Matrix(par.PtSz1,Pn);

% denoising via SSC
[imgCol,Dict]= f_ColFilter(imgCol, par.sigma, DicOri, ...
    par.PtSz1, par.K, par.learnDic);
imgCol = imgCol+ones(size(imgCol,1),1)*vecOfMeans;
imgCol(:,blkOrder(1 : numOfBlks))=imgCol(:,1 : numOfBlks);

% Subimage averaging
count = 1;
Weight= zeros(imgSz);
IMout = zeros(imgSz);
[rows,cols] = ind2sub(imgSz-n+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);
    block =reshape(imgCol(:,count),[n,n]);
    IMout(row:row+n-1,col:col+n-1)=IMout(row:row+n-1,col:col+n-1)+block;
    Weight(row:row+n-1,col:col+n-1)=Weight(row:row+n-1,col:col+n-1)+ones(n);
    count = count+1;
end;

IOut1 = IMout./Weight;


%% Step 2
n=par.PtSz2; % sqrt(n) in Step 2
subImgSz=floor((imgSz-[n n])/par.SlidDt2+[1 1]);   %每个子图的大小
% extract patches
[imgCol,idx] = f_im2col(IOut1,[n n],par.SlidDt2);
% patch ordering
blkOrder=c_PatchSort( imgCol,n,subImgSz );



imgCol=imgCol(:,blkOrder); 
%**************************************************************************
% 全局滤波,denoising via 2-D wavelet hard-thresholding
numOfBlks=size(imgCol,2);
maxBlks=512*512;
numWavFil=floor(numOfBlks/maxBlks);

for k=1:numWavFil
    imgCol(:,k*maxBlks-maxBlks+1:k*maxBlks) = ...
        wdencmp('gbl',imgCol(:,k*maxBlks-maxBlks+1:k*maxBlks),par.waveType,4,par.lamda,'h',1);
end

imgCol(:,numWavFil*maxBlks+1:end) = ...
    wdencmp('gbl',imgCol(:,numWavFil*maxBlks+1:end),par.waveType,4,par.lamda,'h',1);
%**************************************************************************
% 滤波结果排序回原来的顺序
imgCol(:,blkOrder)=imgCol;
% 子图平均, Subimage averaging
count = 1;
Weight= zeros(imgSz);
IMout = zeros(imgSz);
[rows,cols] = ind2sub(imgSz-n+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);
    block =reshape(imgCol(:,count),[n,n]);
    IMout(row:row+n-1,col:col+n-1)=IMout(row:row+n-1,col:col+n-1)+block;
    Weight(row:row+n-1,col:col+n-1)=Weight(row:row+n-1,col:col+n-1)+ones(n);
    count = count+1;
end

IOut2 = IMout./Weight;
% Exponential transformation
IOut1=exp(IOut1(1:IRow,1:ICol));
IOut2=exp(IOut2(1:IRow,1:ICol));



end

% Denoising via SSC
function [ imgColFilter,Dict ] = f_ColFilter( imgCol,errT,DicOri,n,K,learnDic )

if learnDic==0,  Dict=DicOri; end
if learnDic==1    
    par2.K = K;
    par2.numIteration = 5;
    par2.errorFlag = 1; 
    par2.errorGoal = errT*1.15;
    par2.preserveDCAtom = 0; 
    par2.initialDictionary = DicOri(:,1:par2.K );    
    par2.InitializationMethod =  'GivenMatrix';
    par2.maxBlocks=16000;    
    par2.displayProgress = 0;
    
    [Dict,~] = f_SAR_KSVD(imgCol,par2);
%     disp('finished Trainning dictionary');
end

maskFil=zeros(size(imgCol));
imgColFilter=zeros(size(imgCol));
numOfBlks=size(imgCol,2);
blkSD=4;
blkWidth=8;
blkW2SD=blkWidth/blkSD;
blkTimes=ceil(numOfBlks/blkSD);

% 利用mexSOMP算法
tmpBlk=[];
for k=0:blkTimes-blkW2SD-1
    tmpBlk=[tmpBlk,[k*blkSD+1:(k+blkW2SD)*blkSD]];
end
k=blkTimes-blkW2SD;
tmpBlk=[tmpBlk,[k*blkSD+1:numOfBlks]];
tmpBlk=imgCol(:,tmpBlk);

param.L=n*n/2;
param.eps=errT^2*n*n;
ind_groups=int32([0:blkWidth:blkWidth*k]);

Coefs=mexSOMP(tmpBlk,Dict,ind_groups,param);
clear tmpBlk;
for kk=1 : full(max(sum(isnan(Coefs))))
    Coefs(isnan(Coefs))=0;
end

% 重新拼接回
Coefs=Dict*Coefs;
for k=0:blkTimes-blkW2SD-1
    imgColFilter(:,k*blkSD+1:(k+blkW2SD)*blkSD)=...
        imgColFilter(:,k*blkSD+1:(k+blkW2SD)*blkSD)+...
        Coefs(:,k*blkWidth+1:(k+1)*blkWidth);

    maskFil(:,k*blkSD+1:(k+blkW2SD)*blkSD)=...
        maskFil(:,k*blkSD+1:(k+blkW2SD)*blkSD)+1;
end
k=blkTimes-blkW2SD;
imgColFilter(:,k*blkSD+1:end)=imgColFilter(:,k*blkSD+1:end)+...
    Coefs(:,k*blkWidth+1:end);

maskFil(:,k*blkSD+1:end)=...
        maskFil(:,k*blkSD+1:end)+1;
imgColFilter=imgColFilter./maskFil;
end

function [Haar]=Generate_Haar_Matrix(n)

D1=sparse(n,n);
v=sparse([1 zeros(1,n-2), -1]/2);
for k=1:1:n
    D1(k,:)=v;
    v=[v(end),v(1:end-1)];
end;
D2=sparse(n,n);
v=[1 1 zeros(1,n-4), -1 -1]/4;
for k=1:1:n
    D2(k,:)=v;
    v=[v(end),v(1:end-1)];
end;
S1=abs(D1);
S2=abs(D2);
Haar=[kron(S2,S2),kron(S2,D2),kron(D2,S2),kron(D2,D2),...
                             kron(S1,D1),kron(D1,S1),kron(D1,D1)];
end

function [DCT]=Generate_DCT_Matrix(n,Pn)

DCT=zeros(n,Pn);
for k=0:1:Pn-1,
    V=cos([0:1:n-1]'*k*pi/Pn);
    if k>0, V=V-mean(V); end;
    DCT(:,k+1)=V/norm(V);
end;
DCT=kron(DCT,DCT);

end
