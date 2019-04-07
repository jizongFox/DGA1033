clc;
clear all;
rng(1)

%close all;
folderImg = '../PROSTATE/train/Img/';
folderGT = '../PROSTATE/train/GT/';

sizeImg = [255 255 55];
atlasCenter = ceil(sizeImg/2);


% Some GC initialization

diceVals = [];

doCreateAtlas = true;

if doCreateAtlas   
    disp('Generating data...');
    [volumes,masks,centroids,atlas] = prepareData(folderImg, folderGT, sizeImg);
    save('data.mat','volumes','masks','centroids','atlas');
else
    disp('Loading data...');
    load('data.mat');
end

%%

% Image normalization: 0=none, 1=[0,1] range, 2=hist. equilization 
imgNorm = 2;


% For computing the GC binary weights
kernelSize = [3 3 3];
%kernelSize = 5;
kernel = ones(kernelSize(1),kernelSize(2),kernelSize(3));
kernel((kernelSize(1)+1)/2,(kernelSize(2)+1)/2,(kernelSize(3)+1)/2) = 0;
epsW = 1e-6;
sigmaW = 500;

% All pixels with pior below this value are considered BG
minProbThreshold = 0; 

% All pixels with pior above this value are considered FG
maxProbThreshold = 1;

% GC lambda (global value)
lambdaGC = 1000; 

bigConst = 1e6;

% Find atlas positions for min and max threshold
idxAtlasMin = find(atlas > minProbThreshold);
idxAtlasMax = find(atlas >= maxProbThreshold);

[x y z] = ind2sub(sizeImg, idxAtlasMin);
posAtlasMin = [x y z];
[x y z] = ind2sub(sizeImg, idxAtlasMax);
posAtlasMax = [x y z];

% Find atlas bounding box and use it to build prior
% Note: make box 1 pixel larger to have a border of BG seeds
boxMin = min(posAtlasMin)-1; 
boxMax = max(posAtlasMin)+1;

prior = -bigConst*ones(sizeImg);
prior(idxAtlasMin) = atlas(idxAtlasMin);
prior(idxAtlasMax) = bigConst;
prior = prior(boxMin(1):boxMax(1),boxMin(2):boxMax(2),boxMin(3):boxMax(3));

diceVals = [];

%close all;

priorInfo.prior = prior;
priorInfo.centroids = centroids;


% Do 3D segmentation
for i=1:size(volumes,1)
    fprintf('\nSegmenting case %d\n', i);  
    
    mask = squeeze(masks(i,:,:,:));
    
    idxFG = find(mask);
    
    if isempty(idxFG)
        disp('Empty FG, skipping...');
        continue;
    end
    
    vol = squeeze(volumes(i,:,:,:));
    
    % Find corresponding box in image. Clip to avoid going out of image
    imgMin = floor(boxMin + centroids(i,:) - atlasCenter);
    imgMax = floor(boxMax + centroids(i,:) - atlasCenter);
    cropMin = max(imgMin,1);    
    cropMax(1) = min(imgMax(1), sizeImg(1));
    cropMax(2) = min(imgMax(2), sizeImg(2));
    cropMax(3) = min(imgMax(3), sizeImg(3));
        
    volCrop = vol(cropMin(1):cropMax(1),cropMin(2):cropMax(2),cropMin(3):cropMax(3));           
    
    
    if imgNorm == 1
        volCrop = mat2gray(volCrop);
    elseif imgNorm == 2
        volCrop = histeq(volCrop);
    end     
    
    maskCrop = mask(cropMin(1):cropMax(1),cropMin(2):cropMax(2),cropMin(3):cropMax(3));   
    
    priorMin = cropMin - imgMin + 1;
    priorMax = size(prior) + imgMax - cropMax;
    
    priorCrop = prior(priorMin(1):priorMax(1),priorMin(2):priorMax(2),priorMin(3):priorMax(3));
    
    centerSlice = floor(size(maskCrop,3)/2);
    
    I1 = priorCrop(:,:,centerSlice);
    I2 = maskCrop(:,:,centerSlice);
    
    contImg = contourSeg(I1,I2,[1 0 0],1);
            
    figure(1), imshow(contImg)
    %hold on;
    %plot(centroids(i,2),centroids(i,1),'*');
    %hold off;               
       
    
    % Binary potentials
    %W = computeWeights3D(vol, kernel, weightsSigma, weightsEps);    
    N = numel(volCrop);
    
    % Unary potentials
    U = zeros(2,N);    
    U(2,:) = 0.5-priorCrop(:);
        
    weightFile = ['WeightMat/' num2str(sigmaW) '_' num2str(i) '.mat']; 
    
    if exist(weightFile, 'file') == 2
       disp('Loading W matrix...');       
       load(weightFile);
       disp('done.');
   else 
        disp('Computing W matrix...');
        W = computeWeights3D(volCrop, kernel, sigmaW, epsW);        
        save(weightFile, 'W','-v7.3');
        disp('done.');
   end
    
    % Run max flow algorithm
    disp('Running graph-cut...');
    hbk = BK_Create(N,nnz(kernel)*N);
    BK_SetNeighbors(hbk,lambdaGC*W);                   
    BK_SetUnary(hbk,U);
    E = BK_Minimize(hbk);    
    seg = double(BK_GetLabeling(hbk)) - 1;
    BK_Delete(hbk);        
    disp('done.');

    
    segCrop = reshape(seg, size(volCrop));    
        
    contPred = contourSeg(volCrop(:,:,centerSlice),segCrop(:,:,centerSlice),[0 1 0],1); 
    figure(4), imshow(contPred), title('Prediction');               
            
    % Compute Dice
    diceFG = 2*nnz(maskCrop & segCrop)/(nnz(maskCrop) + nnz(segCrop));
    disp(['Dice FG : ' num2str(diceFG)]);  
    
    diceVals(end+1) = diceFG;
    
    disp(['Mean Dice FG : ' num2str(mean(diceVals))]); 
    
    % Generate final segmentation
    segPred = zeros(size(vol),'uint8');
    segPred(cropMin(1):cropMax(1),cropMin(2):cropMax(2),cropMin(3):cropMax(3)) = uint8(segCrop);
    
    priorInfo.cropMin(i,:) = cropMin;
    priorInfo.cropMax(i,:) = cropMax;
    priorInfo.priorMin(i,:) = priorMin;
    priorInfo.priorMax(i,:) = priorMax;    
end

disp(['final Mean Dice FG : ' num2str(mean(diceVals))]);  

save('priorInfo.mat', 'priorInfo');


