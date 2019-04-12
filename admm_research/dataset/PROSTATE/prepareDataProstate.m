function [volumes,masks,centroids,atlas] = prepareDataProstate(folderImg, folderGT, sizeImg, targetClass)

files = dir([folderImg '*.png']);

N = sizeImg(1) * sizeImg(2) * sizeImg(3);

atlasCenter = ceil(sizeImg/2);

volumes = [];
masks = [];

% Load volumes and GT
for f=1:numel(files)    
    fprintf('Processing %s\n', files(f).name);
    
    % Read imag and GT
    img = double(imresize(imread([folderImg files(f).name]),sizeImg(1:2) ,'nearest'))/255;    
    GT = double(imresize(imread([folderGT files(f).name]),sizeImg(1:2),'nearest'));    
    %figure(1), imshow(Img)
    %figure(2), imshow(GT), title('Ground truth');
    
    imgInfo = sscanf(files(f).name,'Case%d_%d_%d.png');     
    
    volumes{imgInfo(1)+1}(:,:,imgInfo(3)+1) = img;
    masks{imgInfo(1)+1}(:,:,imgInfo(3)+1) = GT;
    
    %break;
end


centroids = [];
atlas = zeros(sizeImg);
        
%%
subjectCount = 0;
% Compute centroid and atlas
for i=1:numel(volumes)    
    mask = masks{i};    
    
    % Compute centroid
    idxFG = find(mask==targetClass);    
    
    if isempty(idxFG)
        continue;
    end
    
    vol = volumes{i};    
    
    [x y z] = ind2sub(size(mask),idxFG);
    posFG = [x y z];
    sizeFG = size(idxFG,1);        
    
    cc = mean(posFG);
    centroids(i,:) = cc;
        
    posShift = floor(posFG + atlasCenter-cc);
    atlasIdx = sub2ind(sizeImg,posShift(:,1),posShift(:,2),posShift(:,3));
    atlas(atlasIdx) = atlas(atlasIdx) + 1; 
    subjectCount = subjectCount+1;
    
    % Show current atlas
    %figure(10), imagesc(atlas(:,:,32))
end

atlas = atlas/subjectCount;

