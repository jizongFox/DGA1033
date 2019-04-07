function [volumes,masks,centroids,atlas] = prepareData(folderImg, folderGT, sizeImg)

files = dir([folderImg '*.png']);

N = sizeImg(1) * sizeImg(2) * sizeImg(3);

atlasCenter = ceil(sizeImg/2);

volumes = [];
masks = [];

% Load volumes and GT
for f=1:numel(files)    
    fprintf('Processing %s\n', files(f).name);
    
    % Read imag and GT
    img = single(imresize(imread([folderImg files(f).name]),sizeImg(1:2) ,'nearest'))/255;    
    GT = single(imresize(imread([folderGT files(f).name]),sizeImg(1:2),'nearest'))/255;    
    %figure(1), imshow(Img)
    %figure(2), imshow(GT), title('Ground truth');
    
    imgInfo = sscanf(files(f).name,'Case%d_%d_%d.png');     
    
    volumes(imgInfo(1)+1,:,:,imgInfo(3)+1) = img;
    masks(imgInfo(1)+1,:,:,imgInfo(3)+1) = GT;
    
    %break;
end
% volumes patient, 255, 255, slice_max_num

centroids = [];
atlas = zeros(sizeImg);
        
%%
subjectCount = 0;
% Compute centroid and atlas
for i=1:size(volumes,1)    
    
    mask = squeeze(masks(i,:,:,:));    
    
    % Compute centroid
    idxFG = find(mask);    
    
    if isempty(idxFG)
        continue;
    end
    
    vol = squeeze(volumes(i,:,:,:));    
    
    [x y z] = ind2sub(size(mask),idxFG);
    posFG = [x y z];
    sizeFG = size(idxFG,1);        
    
    centroids(i,:) = mean(posFG);
    
    posShift = floor(posFG + atlasCenter-centroids(i,:));
    atlasIdx = sub2ind(sizeImg,posShift(:,1),posShift(:,2),posShift(:,3));
    atlas(atlasIdx) = atlas(atlasIdx) + 1; 
    subjectCount = subjectCount+1;
    
    % Show current atlas
    %figure(10), imagesc(atlas(:,:,32))
end

atlas = atlas/subjectCount;

