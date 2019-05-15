% mean dice fg: 0.67328 when usePrior = false
% mean dice fg: 0.67352 when usePrior = true

clc;
clear all;
rng(1)

usePrior = false;

addpath(genpath('BK_matlab'));

%close all;
folderImg = '../PROSTATE/train/Img/';
folderGT = '../PROSTATE/train/GT/';
folderPrior = '../PROSTATE/train/prior/';

sizeImg = [256 256 55];
atlasCenter = ceil(sizeImg/2);

targetClass = 255;

dataFile = 'dataProstate.mat';

if exist(dataFile, 'file') == 2
    disp('Loading data...');
    load(dataFile);
else
    disp('Generating data...');
    [volumes,masks,centroids,atlas] = prepareDataProstate(folderImg, folderGT, sizeImg, targetClass);
    save(dataFile,'volumes','masks','centroids','atlas');    
end

%%
savePrior = false;

close all

% Image normalization: 0=none, 1=[0,1] range, 2=hist. equilization 
imgNorm = 0;


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
if usePrior
    lambdaGC = 1000; 
% lambdaGC =0
else
    lambdaGC = 100; 
% lambdaGC = 0 
end

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
if usePrior
    prior(idxAtlasMin) = atlas(idxAtlasMin);
else
    prior(idxAtlasMin) = 0.5;
end
prior(idxAtlasMax) = bigConst;
prior = prior(boxMin(1):boxMax(1),boxMin(2):boxMax(2),boxMin(3):boxMax(3));

diceVals = [];
sizeDiffs = [];

%close all;

priorInfo.prior = prior;
priorInfo.centroids = centroids;



% Do 3D segmentation
for i=1:numel(volumes)
    fprintf('\nSegmenting case %d\n', i);  
    
    if isempty(masks{i})
        continue;
    end
    
    mask = (masks{i}==targetClass);
    
    idxFG = find(mask);
    
    if isempty(idxFG)
        disp('Empty FG, skipping...');
        continue;
    end
    
    vol = volumes{i};
    
    % Find corresponding box in image. Clip to avoid going out of image
    imgMin = floor(boxMin + centroids(i,:) - atlasCenter);
    imgMax = floor(boxMax + centroids(i,:) - atlasCenter);
    cropMin = max(imgMin,1);    
    cropMax(1) = min(imgMax(1), size(vol,1));
    cropMax(2) = min(imgMax(2), size(vol,2));
    cropMax(3) = min(imgMax(3), size(vol,3));
        
    volCrop = vol(cropMin(1):cropMax(1),cropMin(2):cropMax(2),cropMin(3):cropMax(3));           
    
    
    if imgNorm == 1
        volCrop = mat2gray(volCrop);
    elseif imgNorm == 2
        volCrop = histeq(volCrop);
    end     
    
    maskCrop = mask(cropMin(1):cropMax(1),cropMin(2):cropMax(2),cropMin(3):cropMax(3));   
        
    priorMin = cropMin - imgMin + 1;
    priorMax = size(prior) + cropMax - imgMax;  
    
    priorCrop = prior(priorMin(1):priorMax(1),priorMin(2):priorMax(2),priorMin(3):priorMax(3));
    
    centerSlice = floor(centroids(i,3)-cropMin(3)+1);
    
    if savePrior
        priorInt = uint8(255*min(max(priorCrop,0),1));
        priorImg = zeros(size(vol),'uint8');
        priorImg(cropMin(1):cropMax(1),cropMin(2):cropMax(2),cropMin(3):cropMax(3)) = priorInt;        
        contImg = contourSeg(priorImg(:,:,centerSlice), mask(:,:,centerSlice),[1 0 0],1);
        figure(333), imshow(contImg);
        pause(.5);
        
        if i <= 10
            caseName = ['0' num2str(i-1)];
        else
            caseName = num2str(i-1);
        end                
        
        for ss=1:size(priorImg,3)
            if ss <= 10
                sliceName = ['0' num2str(ss-1)];
            else
                sliceName = num2str(ss-1);
            end
        
            fname = [folderPrior 'Case' caseName '_0_' sliceName '.png'];                                        
            %imwrite(priorImg(:,:,ss),fname);
        end
        
        continue;
    end
    
    I1 = priorCrop(:,:,centerSlice);
    I2 = maskCrop(:,:,centerSlice);
    
    contImg = contourSeg(I1,I2,[1 0 0],1);
            
    %figure(1), imshow(contImg)
    %hold on;
    %plot(centroids(i,2),centroids(i,1),'*');
    %hold off;               
       
    
    % Binary potentials
    %W = computeWeights3D(vol, kernel, weightsSigma, weightsEps);    
    N = numel(volCrop);
    
    % Unary potentials
    U = zeros(2,N);   
    U(2,:) = 0.5-priorCrop(:);
        
    weightFile = ['WeightMatProstate/' num2str(sigmaW) '_' num2str(i) '.mat']; 
    
    if exist(weightFile, 'file') == 2
       disp('Loading W matrix...');       
       load(weightFile);
       disp('done.');
   else 
        disp('Computing W matrix...');
        W = computeWeights3D(volCrop, kernel, sigmaW, 0);        
        save(weightFile, 'W','-v7.3');
        disp('done.');
   end
    
    % Run max flow algorithm
    disp('Running graph-cut...');
    ttt = tic;
    hbk = BK_Create(N,nnz(kernel)*N);
    BK_SetNeighbors(hbk,lambdaGC*W);                   
    BK_SetUnary(hbk,U);
    E = BK_Minimize(hbk);    
    seg = double(BK_GetLabeling(hbk)) - 1;
    BK_Delete(hbk);        
    disp('done.');
    toc(ttt)

    
    segCrop = reshape(seg, size(volCrop));    
        
    contPred = contourSeg(volCrop(:,:,centerSlice),segCrop(:,:,centerSlice),[0 1 0],1); 
    %figure(4), imshow(contPred), title('Prediction');
    figure(1)
        subplot(1,2,1), imshow(contPred, 'InitialMagnification', 600), title('Prediction'); 
        subplot(1,2,2), imshow(contImg, 'InitialMagnification', 600);
            
    % Compute Dice
    diceFG = 2*nnz(maskCrop & segCrop)/(nnz(maskCrop) + nnz(segCrop));
    disp(['Dice FG : ' num2str(diceFG)]);  
    
    diceVals(end+1) = diceFG;
    sizeDiffs(end+1) = nnz(maskCrop) - nnz(segCrop);
    
    disp(['Mean Dice FG : ' num2str(mean(diceVals))]); 
    
    % Generate final segmentation
    segPred = zeros(size(vol),'uint8');
    segPred(cropMin(1):cropMax(1),cropMin(2):cropMax(2),cropMin(3):cropMax(3)) = uint8(segCrop);
    
    priorInfo.cropMin(i,:) = cropMin;
    priorInfo.cropMax(i,:) = cropMax;
    priorInfo.priorMin(i,:) = priorMin;
    priorInfo.priorMax(i,:) = priorMax;    
end

figure(100), hist(diceVals);

disp(['final Mean  Dice FG : ' num2str(mean(diceVals))]);  

save('priorInfoProstate.mat', 'priorInfo');

figure(12), scatter(diceVals,abs(sizeDiffs))


