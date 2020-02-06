function IM_out=resizing_cnn(imag)
%% Init
% clear,close all;
sized_hight=200;sized_width=430;
sized=sized_hight*sized_width;
% H=randi([200,700],1);
% W=randi([200,700],1);
%% Read image
imag=(imread(imag));
%% Detection
% imag=detect(imag);
H=size(imag,1);W=size(imag,2);
resized=H*W;
%% Resize to vector
scale=resized/sized;
% IM=imresize(imag(:),[sized,1]);
IM=imresize(imag,floor([H/sqrt(scale) W/sqrt(scale)]));
%% Remove noise;
%seperate channels
r_channel=IM(:,:,1);
b_channel=IM(:,:,2);
g_channel=IM(:,:,3);
%denoise each channel
r_channel=medfilt2(r_channel);
g_channel=medfilt2(g_channel);
b_channel=medfilt2(b_channel);
%restore channels
rgbIM(:,:,1)=r_channel;
rgbIM(:,:,2)=g_channel;
rgbIM(:,:,3)=b_channel;
IM_out=rgbIM;
% Size of the image
img_size = size(rgbIM);
%% Using Kmeans for compression
% A = double(rgbIM);
% 
% A = A / 255; % Divide by 255 so that all values are in the range 0 - 1
% 
% % Reshape the image into an Nx3 matrix where N = number of pixels.
% % Each row will contain the Red, Green and Blue pixel values
% % This gives us our dataset matrix X that we will use K-Means on.
% X = reshape(A, img_size(1) * img_size(2), 3);
% 
% % Run your K-Means algorithm on this data
% % You should try different values of K and max_iters here
% K = 16; 
% epochs = 10;
% 
% % When using K-Means, it is important the initialize the centroids
% % randomly. 
% % You should complete the code in kMeansInitCentroids.m before proceeding
% initial_centroids = kMeansInitCentroids(X, K);
% 
% % Run K-Means
% [centroids, ~] = runkMeans(X, initial_centroids, epochs);
% 
% %% Image Compression 
% %  use the clusters of K-Means to compress an image. 
% 
% % Find closest cluster members
% idx = findClosestCentroids(X, centroids);
% 
% % Essentially, now we have represented the image X as in terms of the
% % indices in idx. 
% 
% % We can now recover the image from the indices (idx) by mapping each pixel
% % (specified by its index in idx) to the centroid value
% X_recovered = centroids(idx,:);
% % Reshape the recovered image into proper dimensions
% IM_out = reshape(X_recovered, img_size(1), img_size(2), 3);
%% Make all images with the same orientation
if img_size(1)> img_size(2)
    IM_out = imrotate(IM_out,-90);
end
%% using pca instead of rgb2gray
% IM_g=         double(rgb2gray(IM));
% A = double(IM_out) ;
% img_size =  size(A);
% X =         reshape(A, img_size(1) * img_size(2), 3);
% mu =        mean(X,1);
% X_norm =    bsxfun(@minus, X, mu);
% sigma =     std(X);
% X_norm =    bsxfun(@rdivide, X_norm, sigma);
% m=          size(X_norm,1);
% sig_cov=    (X_norm'*X_norm)./m;
% [U,S,~]=    svd(sig_cov);
% k=1;
% % den=0;
% % for i=1:length(S)
% %     den=den+S(i,i);
% % end
% % num=0;
% % while(1)
% %     num=num+S(k,k);
% %     if(num / den >=0.93)
% %         break;
% %     else 
% %         k=k+1;
% %     end
% % end
% U_red=      U(:,1:k); % using k=1
% z=          X_norm*U_red;
% IM_out=     reshape(z, img_size(1) , img_size(2),k);
%% Padding
IM_out = padding( IM_out );