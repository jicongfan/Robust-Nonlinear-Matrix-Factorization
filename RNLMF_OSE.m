function [Xr,C,E]=RNLMF_OSE(X,D,lambda,options)
% Out-of-sample extension of RNLMF for denoising
% X:                noisy data matrix, n samples
% D:                dictionary given by the offline RNLMF
% options:          provided by the offline RNLMF
% options.beta:     penalty parameter for ||C||_F^2
% options.lambda:   penalty parameter for E
% options.sigma2:   sigma^2 of Gussian RBF kernel, given by the the offline RNLMF
% options.tol:      tollorence of stopping (default 1e-4)
% options.E_type:   'L1', 'L2', or 'L21' for sparse, Gaussian, or
%                   column-wise sparse corruption (default 'L1')
% options.maxiter:  maximum iterations (default 300)
disp('------Out-of-Sample Extension of RNLMF------')
[m,n]=size(X);
d=size(D,2);
C=zeros(d,n);
E=zeros(m,n);
sigma2=options.sigma2;
beta=options.beta;
lambda=options.lambda;
%
if isfield(options,'E_type')
    E_type=options.E_type;
else
	E_type='L1';
end
%
if isfield(options,'tol')
    tol=options.tol;
else
	tol=1e-5;
end
%
if isfield(options,'maxiter')
    maxiter=options.maxiter;
else
	maxiter=300;
end
%
iter=1;
vC=0;
Kdd=kernel(D,D,sigma2);
Kxx=eye(n);
invKdd=inv(Kdd+beta*eye(d));
J=inf;
while iter<maxiter
    if iter<5
        eta=0;
    else
        eta=0.5;
    end
    iter=iter+1;
    Kdx=kernel(D,X-E,sigma2);
    % update C
    C_new=invKdd*Kdx;   
    % update E
    g_Kxd=-C_new';
    [g_X1,T1,C1]=gXY(g_Kxd,Kdx',X-E,D,sigma2,'X');
    tau=1/sigma2*max(abs(C1(1,:)));
    temp=E-(-(g_X1))/tau;
    %
    switch E_type
        case 'L1'
            E_new=max(0,temp-lambda/tau)+min(0,temp+lambda/tau);
        case 'L2'
            E_new=tau/(lambda+tau)*temp;
        case 'L21'
            E_new=solve_l1l2(temp,lambda/tau);
    end
    %
    if iter<4||mod(iter,50)==0
        dC=norm(C-C_new,'fro')/norm(C,'fro');
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
        dJ=(J-J_new)/J;
        J=J_new;
        else
            J='NotComputed';
            dJ='NotComputed';
        end
        %
        disp(['Iteration ' num2str(iter) ': J=' num2str(J) ', dJ=' num2str(dJ) ', dC=' num2str(dC)...
             ', dE=' num2str(dE)])
    end
    C=C_new;
    E=E_new;
%     if max([dC,dE])<tol||dJ<tol
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
