clc
clear all
warning off
%
load('COIL20_withnoise.mat');
X=X0;% no noise
%%
options.maxiter=200;options.rbf_c=1;
beta_rnlmf=1e-2;lambda_rnlmf=0.05*1e-2;d=16^2;% 0.05* 0.02
tic
[~,E,D_rnlmf,C_rnlmf]=RNLMF(X,d,beta_rnlmf,lambda_rnlmf,options);
toc
e_RNLMF=RNLMF_clustering(C_rnlmf,Label,15,0.01);
