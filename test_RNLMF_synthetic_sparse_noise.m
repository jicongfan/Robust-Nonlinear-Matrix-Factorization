clc
clear all
rng(100)
warning off
K=3;% number of subspaces
m=30;
n0=300;
r=3;
noise_density=0.3;%proportion of corrupted entries
noise_amplitude=1;
for u=1:1
    X=[]
    D_base=randn(m,r);
    for k=1:K
        x=unifrnd(-1,1,[r,n0]);
        T=randn(m,r)*x...
            +1*(randn(m,r)*x.^2+randn(m,1)*[x(1,:).*x(2,:)]++randn(m,1)*[x(1,:).*x(3,:)]+randn(m,1)*[x(2,:).*x(3,:)]...
            +randn(m,r)*x.^3+randn(m,1)*[x(1,:).*x(2,:).*x(3,:)]+randn(m,1)*[x(1,:).^2.*x(2,:)]+randn(m,1)*[x(1,:).^2.*x(3,:)]...
            +randn(m,1)*[x(2,:).^2.*x(1,:)]+randn(m,1)*[x(2,:).^2.*x(3,:)]+randn(m,1)*[x(3,:).^2.*x(1,:)]+randn(m,1)*[x(3,:).^2.*x(2,:)]);
        X=[X T];
    end
%%
[m,n]=size(X);
e=randn(1,m*n)*std(X(:))*noise_amplitude;
e(randperm(m*n,ceil(m*n*(1-noise_density))))=0;
E=reshape(e,size(X));
Xn=X+E;% sparse corruption
%% RPCA
[Xr{1}, E_rpca] = RobustPCA(Xn, 0.1);
%% RNLMF
options.maxiter=500;
options.rbf_c=1;
d=m*2*K;
beta=1*1e-3;
lambda=0.1*1e-3;
[Xr{2},E_rnlmf,D,C,J,~]=RNLMF(Xn,d,beta,lambda,options);
%% compute recovery error
for k=1:length(Xr)
    RE(u,k)=norm(X-Xr{k},'fro')/norm(X,'fro');
end
end
%RE_mean=mean(RE)
