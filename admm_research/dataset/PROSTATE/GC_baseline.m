clc;
close all;
folderImg = '../PROSTATE/val/Img/';
folderGT = '../PROSTATE/val/GT/';
files = dir([folderImg '*.png']);

sizeImg = [255 255];
N = sizeImg(1) * sizeImg(2);

[y,x] = ind2sub(sizeImg,1:N);
pos=[x; y]';

% For computing size bounds
% If useFixedBounds=false, bound is estimate from GT size and eps
% If useFixedBounds=true, bound is taken from maxSizeFG
useFixedBounds = true; 
maxSizeFG = 5000;
% Eps factor generate lower and upper bounds for size
sizeEps = 1;

% Size of labeled FG region
seedsSize = 6;

% If useCentroid=true, the FG seeds will be generated around centroid
% else, they will be generated around a andom point in the FG
useCentroid = false;

% For larger value, pixels near GT centroid have greater chance of being
% selected as seed;
randFactor = 4; 

% Defines the START of the BG band (for estimating the intensity distr)
maskScaleMin = 1.5; 
% Defines the END of the BG band (for estimating the intensity distr)
maskScaleMax = 2; 

% Helps find GMM components when few samples
regularizeGMM = 1e-8;

% For computing the GC binary weights
kernelSize = 3;
%kernelSize = 5;
kernel = ones(kernelSize);
kernel((kernelSize+1)/2,(kernelSize+1)/2) = 0;
weightsEps = 1e-6;
weightsSigma = 250;

% GC lambda (global value)
lambdaGC = 1; 

% Some GC initialization
U = zeros(2,N);
bigConst = 1e6;

% Class prior for FG pixels (higher chance of occurring for unlabeled pixels)
priorFG = .6;

diceVals = [];

for f=1:numel(files)    
    fprintf('\n\nProcessing %s\n', files(f).name);
    
    % Read imag and GT
    img = double(imresize(imread([folderImg files(f).name]),sizeImg,'nearest'))/255;    
    GT = double(imresize(imread([folderGT files(f).name]),sizeImg,'nearest'))/255;    
    %figure(1), imshow(Img)
    %figure(2), imshow(GT), title('Ground truth');   
    
    %pause(1)
    
    % Compute centroid
    FG = find(GT);
    [yy, xx] = find(GT);
    posFG = [xx yy];    
    sizeFG = size(FG,1);
    
    contGT = contourSeg(img,GT,[1 0 0],1); 
    
    figure(3), imshow(contGT), title('GT contour and centroid');        
    
    if isempty(FG)
        continue;
    end                
    
    centroid = mean(posFG); 
    dist = pdist2(pos,centroid);
    
    if ~useCentroid
        
        idx = randsample(1:sizeFG, 1, true, 1./(1+dist(FG).^randFactor));
        centroid = posFG(idx,:);
        dist = pdist2(pos,centroid);
    end        

    %disp(centroid)        
        
    hold on;
    plot(centroid(1),centroid(2),'*');
    hold off;
    pause(.1)

    % Generate FG seeds
    
    dist = reshape(dist,sizeImg);
    maskFG = find(dist <= seedsSize & GT == 1);
    
    %seedCandImg = zeros(sizeImg);
    %seedCandImg(seeds) = 1;
    %figure(10), imagesc(seedCandImg);
    
    % Find outer region for BG seeds
    if useFixedBounds
        maxSize = maxSizeFG;        
    else        
        maxSize = sizeFG*(1+sizeEps);
    end
    
    distMinBG = maskScaleMin*sqrt(maxSize/3.1415);
    distMaxBG = maskScaleMax*sqrt(maxSize/3.1415);
    
    maskBG = find(dist >= distMinBG); 
    bandBG = find(dist >= distMinBG & dist <= distMaxBG);
    
    % Unlabeled pixels
    maskUnlabeled = find(dist < distMinBG);    
    
    distrFG = fitgmdist(img(maskFG), 3, 'Regularize', regularizeGMM);
    distrBG = fitgmdist(img(bandBG), 3, 'Regularize', regularizeGMM);
    
    % Evaluate FG probability  
    unlabeled = img(maskUnlabeled);
    p1 = pdf(distrFG,unlabeled)*priorFG;
    p2 = pdf(distrBG,unlabeled)*(1-priorFG);
    pFG = p1 ./ (p1+p2);
    
    % Show pixel probabilities (FG seeds = 1, BG band = -1)   
    imgProb = GT;
    imgProb(maskUnlabeled) = pFG; % + imgProb(maskUnlabeled);
    %imgProb(FG) = 2;
    %imgProb(maskFG) = 1; % + imgProb(maskUnlabeled);    
    %imgProb(bandBG) = -1;
    figure(6), imagesc(imgProb), title('FG probability');
    
    
    % Run GC using seeds and probabilities
    
    % Binary potentials
    W = computeWeights(img, kernel, weightsSigma, weightsEps);       
    
    % Unary potentials
    U(2,:) = bigConst;
    U(2,maskUnlabeled) = 0.5-pFG;    
    U(2,maskFG) = -bigConst;
    
    % Run max flow algorithm
    hbk = BK_Create(N,nnz(kernel)*N);
    BK_SetNeighbors(hbk,lambdaGC*W);                   
    BK_SetUnary(hbk,U);
    E = BK_Minimize(hbk);    
    seg = double(BK_GetLabeling(hbk)) - 1;
    BK_Delete(hbk);

    % Show prediction
    segImg = reshape(seg, size(img));    
    contPred = contourSeg(img,segImg,[0 1 0],1); 
    figure(4), imshow(contPred), title('Prediction');       
            
    % Compute Dice
    diceFG = 2*nnz(GT & segImg)/(nnz(GT) + nnz(segImg));
    disp(['Dice FG : ' num2str(diceFG)]);  
    
    diceVals(end+1) = diceFG;
    
    disp(['Mean Dice FG : ' num2str(mean(diceVals))]);  
    
    %break;
end


