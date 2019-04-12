clc;
clear all;
rng(1)

addpath(genpath('BK_matlab'));

%close all;
folderImg = '../ACDC-2D-All/train/Img/';
folderGT = '../ACDC-2D-All/train/GT/';

sizeImg = [256 256 45];
atlasCenter = ceil(sizeImg/2);

% LV_Class=255 , RV_Class=85;
% targetClass = 85;
targetClass = 255;

dataFile = 'dataACDC_LV.mat';

if exist(dataFile, 'file') == 2
    disp('Loading data...');
    load(dataFile);
else
    disp('Generating data...');
    [volumes,masks,centroids,atlas] = prepareDataACDC(folderImg, folderGT, sizeImg, targetClass);
    save(dataFile,'volumes','masks','centroids','atlas');    
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

if targetClass == 85 % RV seg
    sigmaW = 5000;
    lambdaGC = 2500; 
elseif targetClass == 255 % LV seg
    sigmaW = 500;
    lambdaGC = 1000;
%     lambdaGC = 0;
end

% All pixels with pior below this value are considered BG
minProbThreshold = 0; 

% All pixels with pior above this value are considered FG
maxProbThreshold = 1;

bigConst = 1e6;

priorInfo = [];

% Do 3D segmentation
for c=1:2
    fprintf('\nSegmenting for cariac cycle %d\n-------------------\n', c);  
    
    % Find atlas positions for min and max threshold
    idxAtlasMin = find(atlas{c} > minProbThreshold);
    idxAtlasMax = find(atlas{c} >= maxProbThreshold);

    [x y z] = ind2sub(sizeImg, idxAtlasMin);
    posAtlasMin = [x y z];
    [x y z] = ind2sub(sizeImg, idxAtlasMax);
    posAtlasMax = [x y z];

    % Find atlas bounding box and use it to build prior
    % Note: make box 1 pixel larger to have a border of BG seeds
    boxMin = min(posAtlasMin)-1; 
    boxMax = max(posAtlasMin)+1;

    prior = -bigConst*ones(sizeImg);
    prior(idxAtlasMin) = atlas{c}(idxAtlasMin);
    prior(idxAtlasMax) = bigConst;    
    
    prior = prior(boxMin(1):boxMax(1),boxMin(2):boxMax(2),boxMin(3):boxMax(3));

    diceVals = [];

    %close all;
    
    cc = squeeze(centroids(:,c,:));

    priorInfo{c}.prior = prior;
    priorInfo{c}.centroids = cc;
    
    for i=1:numel(volumes)
        fprintf('\nSegmenting case %d\n', i);  
        
        if isempty(masks{i})
            continue;
        end
           
        mask = (masks{i}{c}==targetClass);
                
        idxFG = find(mask);

        if isempty(idxFG)
            disp('Empty FG, skipping...');
            continue;
        end       

        vol = volumes{i}{c};
        vol = vol(:,:,2:end);

        % Find corresponding box in image. Clip to avoid going out of image
        imgMin = floor(boxMin + cc(i,:) - atlasCenter);
        imgMax = floor(boxMax + cc(i,:) - atlasCenter);
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

        centerSlice = floor(cc(i,3)-cropMin(3)+1);

        I1 = priorCrop(:,:,centerSlice);
        I2 = maskCrop(:,:,centerSlice);

        contImg = contourSeg(I1,I2,[1 0 0],1);

        figure(1), imshow(contImg)
        %truesize([150 150]);
        %hold on;
        %plot(centroids(i,2),centroids(i,1),'*');
        %hold off;               


        % Binary potentials
        %W = computeWeights3D(vol, kernel, weightsSigma, weightsEps);    
        N = numel(volCrop);

        % Unary potentials
        U = zeros(2,N);    
        U(2,:) = 0.5-priorCrop(:);

        weightFile = ['WeightMatACDC/' num2str(sigmaW) '_' num2str(i) '.mat']; 

        if false %exist(weightFile, 'file') == 2
           disp('Loading W matrix...');       
           load(weightFile);
           disp('done.');
       else 
            %disp('Computing W matrix...');
            W = computeWeights3D(volCrop, kernel, sigmaW, epsW);        
            %save(weightFile, 'W','-v7.3');
            %disp('done.');
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

        priorInfo{c}.cropMin(i,:) = cropMin;
        priorInfo{c}.cropMax(i,:) = cropMax;
        priorInfo{c}.priorMin(i,:) = priorMin;
        priorInfo{c}.priorMax(i,:) = priorMax;  
        
        %TT = segCrop + maskCrop;
        %figure(8), imagesc(TT(:,:,centerSlice));
        %drawnow;
    end
    
    disp(['==== Mean Dice FG : ' num2str(mean(diceVals))]);  
end

save('priorInfoACDC.mat', 'priorInfo');


