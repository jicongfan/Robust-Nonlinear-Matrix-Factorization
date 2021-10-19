function [Xr,E,D,C,options,obj]=RNLMF_faster(X,d,beta,lambda,options)
%%
%%%%%%%%%% Input
% X:                the data matrix, mxn, m featrures and n samples
% d:                dimension of the dictionary D
% beta:             penalty parameter for C
% lambda:           penalty parameter for E
% options.rbf_c:    a constant for estimating the sigma of Gaussian kernel,
%                   e.g. 1, 3,or 5. (default 1)
% options.tol:      tollorence of stopping (default 1e-4)
% options.C_type:   "L2', 'L1', or 'L2' for least-square, sparse, or
%                   low-rank regularization on C (default 'L2')
% options.E_type:   'L1', 'L2', or 'L21' for sparse, Gaussian, or
%                   column-wise sparse corruption (default 'L1')
% options.maxiter:  maximum iterations (default 300)
%%%%%%%%% output
% Xr:       recovered matrix
% E:        noise matrix
% D:        dictionary
% C:        coefficient matrix
% options:  parameter sets (may be used in out-of-sample extension)
%%%%%%
% @ Written by Jicong Fan, 2019.12. 
%%
disp('------Robust Nonlinear Matrix Factorization------')
[m,n]=size(X);
E=zeros(m,n);
% options
if isfield(options,'rbf_c')
    rbf_c=options.rbf_c;
else
	rbf_c=1;
end
disp(['Computing the sigma of Gaussian kernel ......'])
if n<8000
    Xs=X;
else
    Xs=X(:,randperm(n,8000));
end
XX=sum(Xs.*Xs,1);
dist=repmat(XX,size(Xs,2),1) + repmat(XX',1,size(Xs,2)) - 2*Xs'*Xs;
sigma2=(mean(real(dist(:).^0.5))*rbf_c)^2;
disp(['sigma=' num2str(sqrt(sigma2))])
if isempty(d)
    Kxx=kernel(Xs,Xs,sigma2);
    [U,S,V]=svd(Kxx);
    S=diag(S);
    figure
    S=cumsum(S)/sum(S);
    bar(S)
    d=find(S>0.99,1);
    disp(['The estimated d is ' num2str(d)])
end
clear XX dist Xs Kxx U S V
C=zeros(d,n);
D=randn(m,d);
%
if isfield(options,'tol')
    tol=options.tol;
else
	tol=1e-4;
end
%
if isfield(options,'E_type')
    E_type=options.E_type;
else
	E_type='L1';
end
%
if isfield(options,'C_type')
    C_type=options.C_type;
else
	C_type='L2';
end
%
if isfield(options,'maxiter')
    maxiter=options.maxiter;
else
	maxiter=300;
end
if isfield(options,'compute_obj')
    compute_obj=options.compute_obj;
else
	compute_obj=0;
end
%
if isfield(options,'eta')
    v_eta=options.eta;
else
	v_eta=0.5;
end
% save options for out-of-sampe extension
options.tol=tol;
options.E_type=E_type;
options.C_type=C_type;
options.maxiter=maxiter;
options.lambda=lambda;
options.sigma2=sigma2;
options.beta=beta;
options.eta=v_eta;
options
%
iter=0;
vD=0;
if n<8000
    Kxx=eye(n);
end
J=inf;
obj=[];
while iter<maxiter
    if iter<5
        eta=0;
    else
        eta=v_eta;
    end
    iter=iter+1;
    %
    Kdx=kernel(D,X-E,sigma2);
    Kdd=kernel(D,D,sigma2);
    
    % update C
    switch C_type
        case 'L1'
            if iter<50
                C_new=inv(Kdd+beta*eye(d))*Kdx;
            else
            tau=1*normest(Kdd);
%             temp=C-(-Kdx+Kdd*C)/tau;
%             C_new=max(0,temp-beta/tau)+min(0,temp+beta/tau);
            Ct=C;
            for jj=1:5
            temp=Ct-(-Kdx+Kdd*Ct)/tau;
            C_new=max(0,temp-beta/tau)+min(0,temp+beta/tau);
            Ct=C_new;
            end
            end
        case 'L2'
            C_new=inv(Kdd+beta*eye(d))*Kdx;
        case 'LR'
            if iter<50
                C_new=inv(Kdd+beta*eye(d))*Kdx;
            else
            tau=1*normest(Kdd);
            Ct=C;
            for jj=1:1
                temp=Ct-(-Kdx+Kdd*Ct)/tau;
                C_new=SVT(temp,tau/beta);
                Ct=C_new;
            end
            end
    end
   
    % update D
    g_Kxd=-C_new';
    g_Kdd=0.5*C_new*C_new';
    [g_D1,T1,C1]=gXY(g_Kxd,Kdx',X-E,D,sigma2,'Y');
    [g_D2,T2,C2]=gXX(g_Kdd,Kdd,D,sigma2);
    tau=1/sigma2*(2*T2-diag(C1(1,:)+2*C2(1,:))); 
    % tau=normest(tau);
    g_D=(g_D1+g_D2)/tau;
    vD=eta*vD+g_D;
    D_new=D-vD;

    % update E
    Kdx=kernel(D_new,X-E,sigma2);
    [g_X1,T1,C1]=gXY(g_Kxd,Kdx',X-E,D_new,sigma2,'X');
    tau=1/sigma2*max(abs(C1(1,:)));
    temp=E-(-(g_X1))/tau;
    switch E_type
        case 'L1'
            E_new=max(0,temp-lambda/tau)+min(0,temp+lambda/tau);
        case 'L2'
            E_new=tau/(lambda+tau)*temp;
        case 'L21'
            E_new=solve_l1l2(temp,lambda/tau);
    end
    %
    if iter<4||mod(iter,50)==0||compute_obj
        dC=norm(C-C_new,'fro')/norm(C,'fro');
        dD=norm(D-D_new,'fro')/norm(D,'fro');
        dE=norm(E-E_new,'fro')/norm(E,'fro');
        %
        if n<8000
        switch E_type
            case 'L1'
                J_new=0.5*trace(Kxx-C'*Kdx-Kdx'*C+C'*Kdd*C)+lambda*sum(abs(E(:)))+0.5*beta*sum(C(:).^2);
            case 'L2'
                J_new=0.5*trace(Kxx-C'*Kdx-Kdx'*C+C'*Kdd*C)+0.5*lambda*sum(E(:).^2)+0.5*beta*sum(C(:).^2);
            case 'L21'
                J_new=0.5*trace(Kxx-C'*Kdx-Kdx'*C+C'*Kdd*C)+0.5*lambda*sum(E(:).^2)+0.5*beta*sum(C(:).^2);    
        end
        obj(iter)=J_new;
        dJ=(J-J_new)/J;
        J=J_new;
        else
            J='NotComputed';
            dJ='NotComputed';
        end
        %
        disp(['Iteration ' num2str(iter) ': J=' num2str(J) ', dJ=' num2str(dJ) ', dC=' num2str(dC)...
            ', dD=' num2str(dD) ', dE=' num2str(dE)])
    end
    C=C_new;
    D=D_new;
    E=E_new;
%     if max([dC,dD,dE])<tol||dJ<tol
%         disp('Converged!')
%         break
%     end
end
Xr=X-E;
end

%%
function [K,XY]=kernel(X,Y,sigma2)
nx=size(X,2);
ny=size(Y,2);
XY=X'*Y;
xx=sum(X.*X,1);
yy=sum(Y.*Y,1);
D=repmat(xx',1,ny) + repmat(yy,nx,1) - 2*XY;
K=exp(-D/2/sigma2); 
end
%%
function [g,T,C]=gXY(g_Kxd,Kxd,X,D,sigma2,v)
switch v
    case 'Y'
        T=g_Kxd.*Kxd;% n x d
        C=repmat(sum(T),size(X,1),1);
        g=1/sigma2*(X*T-D.*C);  
    case 'X'
        T=g_Kxd'.*Kxd';% d x n;
        C=repmat(sum(T),size(X,1),1);
        g=1/sigma2*(D*T-X.*C);
end
end

%%
function [g,T,C]=gXX(g_Kdd,Kdd,D,sigma2,I)
if ~exist('I')
    T=g_Kdd.*Kdd;
    C=repmat(sum(T),size(D,1),1);
    g=2/sigma2*(D*T-D.*C);
else
    T=g_Kdd.*Kdd;
    C=repmat(sum(T),size(D,1),1);
    g=2/sigma2*(D.*repmat(diag(T)',size(D,1),1)-D.*C);
end
end 
%%
function [g,T,C]=gXXI(c,K_diag,X,sigma2)
T=c*K_diag;
C=repmat(T,size(X,1),1);
g=2/sigma2*(X.*C-X.*C);
end
%%
function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end
end
%%
function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end

%%
function [Z]=SVT(M,mu)
    [U,sigma,V] = svd(M,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    Z=U(:,1:svp)*diag(sigma)*V(:,1:svp)';
end

