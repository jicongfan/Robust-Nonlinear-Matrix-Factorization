clc
clear all
warning off
%load('mocap_56_06.mat');
load('mocap_01_09.mat');
X0=X;
[m,n]=size(X0);
n_rep=10;
for j=1:n_rep
    %
    Xn{j}=[];
    E0=randn(m,n)*std(X0(:));
    rho=0.1;% noise density 0.1 or 0.3
    E0(randperm(m*n,round(m*n*(1-rho))))=0;
    Xn{j}=X0+E0;
    id_train{j}=randperm(n,round(n*0.5));
    id_test{j}=setdiff(1:n,id_train{j});
    Xn_train{j}=Xn{j}(:,id_train{j});
    Xn_test{j}=Xn{j}(:,id_test{j});
end
Xr={};
options.maxiter=300;options.rbf_c=0.5;
beta_rnlmf=1e-2; lambda_rnlmf=5*1e-5; % or 4,5*1e-5
d=m;
for j=1:n_rep
% RPCA
lambda_rpca=1.5/sqrt(n/2);
[Xr_train{j}{1}, E_rpca] = RobustPCA(Xn_train{j},lambda_rpca,lambda_rpca,1e-5,500);   
[Xr_test{j}{1}, C_rpca,E_rpca] = RobustPCA_OSE(Xn_test{j},Xr_train{j}{1},10,0.001,5); % 20 or 10; 0.001; 5;
% RNLMF
[Xr_train{j}{2},E_rnlmf,D,C,options_train]=RNLMF(Xn_train{j},d,beta_rnlmf,lambda_rnlmf,options);
[Xr_test{j}{2},C_t,E_t]=RNLMF_OSE(Xn_test{j},D,lambda_rnlmf,options_train);
%
end
%%
for j=1:n_rep
    for k=1:2
    RMSE_train(j,k)=norm(X0(:,id_train{j})-Xr_train{j}{k},'fro')/norm(X0(:,id_train{j}),'fro')*100;    
    MAE_train(j,k)=sum(sum(abs(X0(:,id_train{j})-Xr_train{j}{k})))/sum(sum(abs(X0(:,id_train{j}))))*100;
    RMSE_test(j,k)=norm(X0(:,id_test{j})-Xr_test{j}{k},'fro')/norm(X0(:,id_test{j}),'fro')*100;    
    MAE_test(j,k)=sum(sum(abs(X0(:,id_test{j})-Xr_test{j}{k})))/sum(sum(abs(X0(:,id_test{j}))))*100;
    end
end
mean(RMSE_train)
std(RMSE_train)
mean(RMSE_test)
std(RMSE_test)
mean(MAE_train)
std(MAE_train)
mean(MAE_test)
std(MAE_test)

