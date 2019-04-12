function [volumes,masks,centroids,atlas] = prepareDataACDC(folderImg, folderGT, sizeImg, targetClass)

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
           
    imgInfo = sscanf(files(f).name,'patient%d_%d_%d.png'); 
    
    if imgInfo(2) == 1
        volumes{imgInfo(1)+1}{1}(:,:,imgInfo(3)+1) = img;
        masks{imgInfo(1)+1}{1}(:,:,imgInfo(3)+1) = GT;
    else
        volumes{imgInfo(1)+1}{2}(:,:,imgInfo(3)+1) = img;
        masks{imgInfo(1)+1}{2}(:,:,imgInfo(3)+1) = GT;
    end
end
%%

centroids = [];
atlas = [];
        
% Compute centroid and atlas
for c=1:2
    atlas{c} = zeros(sizeImg);
    
    subjectCount = 0;
    
    for i=1:numel(volumes)                  
        
        if isempty(volumes{i})
            continue;
        end
        
        mask = masks{i}{c};    

        % Compute centroid
        idxFG = find(mask==targetClass);    

        if isempty(idxFG)
            continue;
        end

        vol = volumes{i}{c};    

        [x y z] = ind2sub(size(mask),idxFG);
        posFG = [x y z];
        sizeFG = size(idxFG,1);   
        
        cc = mean(posFG);
        centroids(i,c,:) = cc;

        posShift = floor(posFG + atlasCenter-cc);
        atlasIdx = sub2ind(sizeImg,posShift(:,1),posShift(:,2),posShift(:,3));
        atlas{c}(atlasIdx) = atlas{c}(atlasIdx) + 1; 
        subjectCount = subjectCount+1;

        % Show current atlas
        figure(10), imagesc(atlas{c}(:,:,atlasCenter(3)))
    end

    atlas{c} = atlas{c}/subjectCount;
end
