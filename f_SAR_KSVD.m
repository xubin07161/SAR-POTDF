function [Dictionary,output] = f_SAR_KSVD(...
    Data,... % an nXN matrix that contins N signals (Y), each of dimension n.
    param)

Dictionary=param.initialDictionary;

%normalize the dictionary.
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
Dictionary = Dictionary.*repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.
totalErr = zeros(1,param.numIteration);

% the K-SVD algorithm starts here.
blkSSCWidth=8;
blkSSCTime=ceil(size(Data,2)/blkSSCWidth);
if size(Data,2)>param.maxBlocks
    blkSSCTime2=ceil(param.maxBlocks/blkSSCWidth);
    tmp=randperm(blkSSCTime-1);
    tmp=tmp(1:blkSSCTime2);
    tmp2=[];
    for k=1 : blkSSCWidth
        tmp2=[tmp2,(tmp-1)*blkSSCWidth+k];
    end
    tmp2=sort(tmp2);
    Data=Data(:,tmp2);
    ind_groups=int32([0:blkSSCWidth:blkSSCWidth*(blkSSCTime2-1)]);
else
    ind_groups=int32([0:blkSSCWidth:blkSSCWidth*(blkSSCTime-1)]);
end

paraMex.L=blkSSCWidth;
paraMex.eps=(param.errorGoal)^2*size(Dictionary,1);


for iterNum = 1:param.numIteration
    
    CoefMatrix=mexSOMP(Data,Dictionary,ind_groups,paraMex);
    
    for kk=1 : full(max(sum(isnan(CoefMatrix))))
        CoefMatrix(isnan(CoefMatrix))=0;
    end
    
%     replacedVectorCounter = 0;
	rPerm = randperm(size(Dictionary,2));
    for j = rPerm
        [Dictionary(:,j),CoefMatrix] = I_findBetDictElem(Data,...
            Dictionary,j,CoefMatrix);
    end

    if (iterNum>1 & param.displayProgress)
        disp(['Iteration   ',num2str(iterNum),'   Average number of coefficients: ',...
            num2str(length(find(CoefMatrix))/size(Data,2))]);        
    end    
    Dictionary = I_clearDictionary(Dictionary,CoefMatrix(1:end,:),Data);
end

output.CoefMatrix = CoefMatrix;
Dictionary = Dictionary;
%==========================================================================
%  findBetterDictionaryElement

function [betDictElem,CoefMatrix] = I_findBetDictElem(Data,Dictionary,j,CoefMatrix)
relevantDataIndices = find(CoefMatrix(j,:)); 
if (length(relevantDataIndices)<1) 
%     [~,i]=max(sum((Data-Dictionary*CoefMatrix).^2));
    i=1;
    betDictElem = Data(:,i);
    betDictElem = sign(betDictElem(1))*betDictElem./sqrt(betDictElem'*betDictElem);
    CoefMatrix(j,:) = 0;
%     NewVectorAdded = 1;
    return;
end

tmpCoefMatrix = CoefMatrix(:,relevantDataIndices); 
tmpCoefMatrix(j,:) = 0;
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); 

% [betterDictionaryElement,singularValue,betaVector] = svds(errors,1);
[betDictElem,singularValue,betaVector] = f_svdsMax(errors);
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';

%==========================================================================
%  I_clearDictionary
function Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data)
T2 = 0.99;
T1 = 3;
K=size(Dictionary,2);
Er=sum((Data-Dictionary*CoefMatrix).^2,1); % remove identical atoms
tmp=logical(diag(ones(1,K)));
G=Dictionary'*Dictionary; G(tmp)=0;%G = G-diag(diag(G));
for jj=1:K
    if max(G(jj,:))>T2 || length(find(abs(CoefMatrix(jj,:))>1e-7))<=T1 ,
        [~,pos]=max(Er);
        Er(pos(1))=0;
        Dictionary(:,jj)=Data(:,pos(1))/norm(Data(:,pos(1)));
        G=Dictionary'*Dictionary;
        G(tmp)=0;%G = G-diag(diag(G));
    end;
end;

