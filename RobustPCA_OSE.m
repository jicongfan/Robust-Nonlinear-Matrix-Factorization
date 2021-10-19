function [Xr,Z,E]=RobustPCA_OSE(X,X_train,d,beta,lambda,E_regul,maxiter)
[m,n]=size(X);
[U,S,V]=svd(X_train,'econ');
if d<1
    s=diag(S);
    r=cumsum(s)/sum(s);
    d=find(r>0.99,1);
    disp(['The number of PCs is ' num2str(d)])
end
A=U(:,1:d);
Z=zeros(d,n);
E=zeros(m,n);
%
if ~exist('E_regul')
	E_regul='L1';
end
if ~exist('maxiter')
	maxiter=1000;
end
e=1e-5;
%
iter=0;
%
while iter<maxiter
    iter=iter+1;
    % Z_new min ||X-AZ||+beta||Z||
    Z_new=inv(A'*A+beta*eye(d))*(A'*(X-E)); 
    % E_new
    AZ=A*Z_new;
    %
    temp=X-AZ;
    switch E_regul
        case 'L1'
            E_new=max(0,temp-lambda)+min(0,temp+lambda);
        case 'L21'
            E_new=solve_l1l2(temp,lambda);
    end
    %
    %
    J(iter)=0.5*sum((X(:)-AZ(:)-E_new(:)).^2)+0.5*beta*sum(Z_new(:).^2)+lambda*sum(abs(E_new(:)));
    %
    et=[norm(Z_new-Z,'fro')/norm(Z,'fro') norm(E_new-E,'fro')/norm(E,'fro')];
    stopC=max(et);
    %
    isstopC=stopC<e||et(2)<e/10;
    if mod(iter,100)==0||isstopC||iter<=10
        disp(['iteration=' num2str(iter) '/' num2str(maxiter) '  J=' num2str(J(iter)) '  stopC=' num2str(stopC) '  e_E=' num2str(et(2)) ])
    end
    if isstopC
        disp('converged')
        break;
    end
    Z=Z_new;
    E=E_new;
  
end
Xr=X-E;
end
%%
function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end

