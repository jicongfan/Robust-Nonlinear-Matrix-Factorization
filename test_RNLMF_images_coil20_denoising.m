clc
clear all
warning off
%
load('COIL20.mat');
nsbj=20;
N=72*nsbj;
X=fea(1:N,:)';
c1=32;c2=32;
Label=gnd;
%
c=20;
d=16^2;
% Label=Label(1:N)';
for pp=1:1
    id_noisy1=randperm(N,round(N*0.3));% proportion of images with noise
    id_noisy2=randperm(N,round(N*0.3));%
    %
    for i=1:N
        temp=X(:,i);
        img=reshape(temp,c1,c2);
        img=imresize(img,[c,c]);
        Y0(:,i)=img(:);
        if length(intersect(i,id_noisy1))==1 % random
            img=imnoise(img,'salt & pepper',0.25);
        end
%        if length(intersect(i,id_noisy2))==1 % ?Occlusion
%          [a,b]=size(img);r=0.25;w1=round(a*r);w2=round(b*r);
%          p1=randperm(a-w1);p1=p1(1);p2=randperm(b-w2);p2=p2(1);
%         img(p1:p1+w1,p2:p2+w2)=0;
%        end
        Yn(:,i)=img(:);
    end

k=0;
nb=5;
%% robust pca 
lambda_rpca=1/sqrt(N); % need to tune
[Yr{1}, E_rpca] = RobustPCA(Yn, lambda_rpca,lambda_rpca*10,1e-5,500);
[U,~,~]=svd(Yr{1},'econ');D{1}=U(:,1:min(d,size(Yn,1)));
%% RNLMF
options.maxiter=200;options.rbf_c=1;
beta_rnlmf=1e-2;
lambda_rnlmf=0.05*1e-2;% % need to tune
[Yr_rnlmf,E,D_rnlmf,C_rnlmf]=RNLMF(Yn,d,beta_rnlmf,lambda_rnlmf,options);
Yr{2}=Yr_rnlmf;D{2}=D_rnlmf;
%%
for ii=1:length(Yr)
    error_rel(pp,ii)=norm(Y0-Yr{ii},'fro')/norm(Y0,'fro');
end
norm(Y0-Yn,'fro')/norm(Y0,'fro')
sum(abs(Y0(:)-Yn(:)))/sum(abs(Y0(:)))
end
mean(error_rel)*100
std(error_rel)*100
%%
[c1,c2]=size(img);
figure
id=sort(id_noisy1([1:15]*20-10));
nr=length(id);
nc=length(Yr)+2;
Z=zeros(nc*c1,nr*c2);
for i=1:nr
    for j=1:nc
        if j==1
            Dt=reshape(Y0(:,id(i)),[c1 c2]);
        end
        if j==2
            Dt=reshape(Yn(:,id(i)),[c1 c2]);
        end
        if j>2
            Dt=reshape(Yr{j-2}(:,id(i)),[c1 c2]);
        end
        Z((j-1)*c2+1:j*c2,(i-1)*c1+1:i*c1)=Dt;
    end
end
subplot(1,1,1);imshow(Z)


