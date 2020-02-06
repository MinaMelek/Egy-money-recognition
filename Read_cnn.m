clear,close all
% tic 
direct='.\small_50_200';% put your directorry here
fnames=     [dir([direct '\*.jpg']);dir([direct '\*.png']);dir([direct '\*.JPEG']);dir([direct '\*.bmp'])];
numfids=    length(fnames);
% for N=  1:numfids
%     FN=[direct '\' fnames(N).name];
%     x=fnames(N).name;
%     a=find(x=='.');
%     if length(a)>1
%         a=a(end);
%     end
%     im=    resizing_cnn(FN);
% %     s=size(im);rm=0;cm=0;
% %     if s(1)>rm
% %         rm=s(1);
% %     end
% %     if s(2)>cm
% %         cm=s(2);
% %     end
%     imwrite(im,(['small_100\' x(1:a-1) '_s' x(a:end)]),x(a+1:end))
% end
% toc
s=size(permute(imread([direct '\' fnames(1).name]),[3 1 2]));
DataCell_1=zeros([numfids s],'uint8');
Labels_1=zeros(numfids,1);
for N=  1:numfids
    FN =    imread([direct '\' fnames(N).name]);
    FN =    permute(FN,[3 1 2]);
    DataCell_1(N,:,:,:)=FN;
    Labels_1(N)=str2double(fnames(N).name(1));
end
% P =         randperm(numfids);
% DataCell =  DataCell(P,:,:,:);
% Labels =    Labels(P)-1;
clearvars -except D* L*
save('ImageData')