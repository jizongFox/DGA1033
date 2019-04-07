% NOTE: Kernel is a matrix of binary values representing the neighbors
%
% Example: For a 3x3 neighborhood
%
% Kernel =
%   1     1     1
%   1     0     1
%   1     1     1
%
function [M,X] = computeWeights3D(Image, Kernel, sigmaW, epsW)

[W,H,D] = size(Image);

X = Image(:);

[KW,KH,KD] = size(Kernel);
K = nnz(Kernel);

N = size(X,1);
A = padarray(reshape(1:N,W,H,D),[ceil(0.5*(KW-1)) ceil(0.5*(KH-1)) ceil(0.5*(KD-1))]);
Neigh = zeros(N,K);

k = 1;
for i=1:KW
    for j=1:KH  
        for d=1:KD 
            if Kernel(i,j,d) == 0
                continue;
            end

           T = A(i:i+W-1, j:j+H-1, d:d+D-1);       
           Neigh(:,k) = T(:);
           k = k+1;
        end
    end
end

T1 = repmat((1:N)', K, 1);
T2 = Neigh(:);
Z = (T1 > T2);
T1(Z) = [];
T2(Z) = [];

M = sparse(T1, T2, (1-epsW)*exp(-sigmaW*((X(T1,:)-X(T2,:)).^2)) + epsW, N, N);
M = M + M';

%toc
