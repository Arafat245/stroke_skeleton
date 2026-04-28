function [pDP, dist, dist_align,gam]=Align2SequencesDist(T1,T2,T,Ref_c)
%T1 -- reference /////////////////// T2 -- Test
%Ref_c=importdata('B.mat'); -- Anuj Edit
dist=0;dist_align=0;
%Generate Trajectories
k=1;
[ss1, ss2, ss3] = size(T1); %% Anuj Edit
 alp1=zeros(ss2,ss1,ss3);alp2=zeros(ss2,ss1,ss3);
for i=1:T 
    X=squeeze(T1(:,i,:));
    Y=squeeze(T2(:,i,:));
    X0=CenteredScaled(X);
    Y0=CenteredScaled(Y);
    if (norm(X0,'fro')~=0)&(norm(Y0,'fro')~=0)&(~isnan(norm(X0,'fro'))) & (~isnan(norm(Y0,'fro')))
        [dd,Y0_T,tr] = procrustes(X0,Y0,'reflection',0);
%         dist(i)=dd;
%         Y0_T=Y0*tr.T;
        Y0_T=Y0;%*tr.T;
        dist(i)=acos(trace(X0'*Y0_T));
        alp1(k,:,:)=X0;  
        alp2(k,:,:)=Y0_T;  
%     else
%         alp1(k,:,:)=X0;  
%         alp2(k,:,:)=Y0;
     end
    k=k+1;
end
k=k-1;

F_OrtNorm=Form_Basis_New(Ref_c);%Each column is a basis
n=size(Ref_c,1);
%Compute vectors of coefficient of TSRVF w.r.t. the basis F_OrtNorm

h1=Path_to_h(alp1,Ref_c);
h2=Path_to_h(alp2,Ref_c);

for i=1:k
    h1_idx=reshape(squeeze(h1(i,:,:)),(n-1)*3,1);
    h2_idx=reshape(squeeze(h2(i,:,:)),(n-1)*3,1);
    for j=1:size(F_OrtNorm,2)
        coef_h1(j,i)=InnerProd_Q(h1_idx,F_OrtNorm(:,j));
        coef_h2(j,i)=InnerProd_Q(h2_idx,F_OrtNorm(:,j));
    end
end

[gam] = DynamicProgrammingQ_Adam(coef_h1,coef_h2,0,0);
%gamI = invertGamma(gam);
%gamI = (gamI-gamI(1))/(gamI(end)-gamI(1));

clear p p2;
N=20;
for i=1:k
    p=squeeze(alp1(i,:,:))';
    for j=1:N        
        p1(j,i)=p(1,j);
        p1(j+N,i)=p(2,j);
        p1(j+2*N,i)=p(3,j);
    end
end

%""""""""""" p2L = (p2)o(gamI)
p1L = Group_Action_by_Gamma(p1,gam);
for t=1:k
    pDP(:,t,:)=reshape(p1L(:,t),N,3);
end

%%%%%%%%%%%%%%%%%% Compute distance after alignement
k=1;
% alp1=zeros(100,20,3);alp2=zeros(100,20,3);
for i=1:T 
    X=squeeze(pDP(:,i,:));
    Y=squeeze(T2(:,i,:));
    X0=CenteredScaled(X);
    Y0=CenteredScaled(Y);
    if (norm(X0,'fro')~=0)&(norm(Y0,'fro')~=0)&(~isnan(norm(X0,'fro'))) & (~isnan(norm(Y0,'fro')))
        [dd,Y0_T,tr] = procrustes(X0,Y0);
%         dist_align(i)=dd;
%         Y0_T=Y0*tr.T;
        Y0_T=Y0;%*tr.T;
        dist_align(i)=acos(trace(X0'*Y0_T));
        alp1(k,:,:)=X0;  
        alp2(k,:,:)=Y0_T;  
%     else
%         alp1(k,:,:)=X0;  
%         alp2(k,:,:)=Y0;
     end
    k=k+1;
end

