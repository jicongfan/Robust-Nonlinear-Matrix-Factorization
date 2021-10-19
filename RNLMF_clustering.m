function [e]=RNLMF_clustering(C,L,s,lambda)
n=size(C,2);
disp('Constructing affinity matrix ...')
d=size(C,1);
%A=abs(inv(C'*C+lambda*eye(n))*C'*C);
A=abs(1/lambda*C'*inv(eye(d)+1/lambda*C*C')*C);
A=A-diag(diag(A));
for i=1:n
    temp=A(:,i);
    z=sort(temp,'descend');
    A(temp<z(s),i)=0;
    A(:,i)=A(:,i)/sum(A(:,i));
end
A=(A+A')/2;
k=length(unique(L));
A=sparse(A);
disp('Performing spectral clustering ...')
Lr = SpectralClustering(A,k);%,'Eig_Solver', 'eigs');
Lr = bestMap(L,Lr);
e = sum(L(:) ~= Lr(:)) / length(L);
end