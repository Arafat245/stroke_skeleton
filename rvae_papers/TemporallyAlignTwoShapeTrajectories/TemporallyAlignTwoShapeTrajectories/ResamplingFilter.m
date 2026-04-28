function alpha_C_new=ResamplingFilter(skeleton1,s)

[n1,m1,k1]=size(skeleton1);
% tic
%""""""""""" Step the parameter to quantize the sequence
Step=1;
%"""""""""" the sequence length
n=floor(m1/Step);
 tau=zeros(n)';
for i=1:n %for each frame
    X=squeeze(skeleton1(:,Step*i,1:3));
    X0=CenteredScaled(X);
    alpha_C(:,i,:) = X0;
    tau(i)=i/(n-1)-1/(n-1);
end

tau;

%%%%%%%%%%%%%%      RESAMPLING
for i=1:length(s)-1
    k=find(s(i)<=tau(:));
    ind1=k(1)-1;
    ind2=k(1);
    if(ind1==0)
        ind1=1;ind2=2;
    end
%     tau(ind1)
%     s(i)
%     tau(ind2)
% """""""""" tau(ind1)<s(i)<tau(ind2) 
w1=(s(i)-tau(ind1))/(tau(ind2)-tau(ind1));
w2=(tau(ind2)-s(i))/(tau(ind2)-tau(ind1));

 X_new=squeeze(alpha_C(:,ind1,:));
 Y_new=squeeze(alpha_C(:,ind2,:));
 
 w1+w2;
 
 if (norm(X_new,'fro')~=0)&(norm(Y_new,'fro')~=0) &(~isnan(norm(X_new,'fro')))&(~isnan(norm(Y_new,'fro')))
     [d,Y0_newT,tr] = procrustes(X_new,Y_new);
     Y0_newT=Y_new*tr.T;
 else
    Y0_newT=Y_new;
end
 
 theta=acos(trace(X_new*Y0_newT'));
 alpha_C_new(:,i,:)=(1/sin(theta))*(sin(w2*theta)*X_new+sin(w1*theta)*Y0_newT);
end

