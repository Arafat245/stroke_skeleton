function h = Path_to_h(alp0,Ref_c)
% size(Ref_c)
[T,n,dim]=size(alp0);

%Sub-matrix H
for j=1:n-1
    h_j=-1/sqrt(j*(j+1));
    H(j,:)=[repmat(h_j,1,j),-j*h_j,zeros(1,n-j-1)];
end
% size(H)
%Pre-shape space
for i=1:T
    ptmp=H*squeeze(alp0(i,:,:));
    alp(i,:,:)=ptmp/sqrt(InnerProd_Q(ptmp,ptmp));
end
C0=H*Ref_c; 
C=C0/sqrt(InnerProd_Q(C0,C0));%Pre-shape space C



h=zeros(T,n-1,dim);

for i=1:T-1
    X=squeeze(alp(i,:,:));
    Y=squeeze(alp(i+1,:,:));
    if (norm(X,'fro')~=0)&(norm(Y,'fro')~=0)&(~isnan(norm(X,'fro'))) & (~isnan(norm(Y,'fro')))
        [d,Y_new,B] = procrustes(X,Y);
        Y_new=Y*B.T;
     else
         Y_new=ones(31,1);
    end
    V = T*InverseExp(X,Y_new);
    Vnorm=sqrt(InnerProd_Q(V,V));
    Vpar=ParallelTransportNew(X,V,C);
    h(i,:,:)=Vpar/sqrt(Vnorm);
end
V_end=ParallelTransportNew(X,V,Y);
Vnorm=sqrt(InnerProd_Q(V_end,V_end));
Vpar=ParallelTransportNew(Y,V_end,C);
h(T,:,:)=Vpar/sqrt(Vnorm);

