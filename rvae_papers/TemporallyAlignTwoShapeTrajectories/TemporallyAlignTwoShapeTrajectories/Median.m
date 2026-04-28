function Med=Median(O)
%-------- Med is a frame, w is the sliding window
[m,n,p]=size(O);
Med=squeeze(O(1,:,:)); %the initial median
MuV=1;
eps1=0.001;eps2=0.1;
j=1;
dist=[0.01];
while(norm(MuV,'fro')>eps1)  
    for i=2:m
        %Procrustes
        X=squeeze(O(i,:,:));
        if (norm(X,'fro')~=0)&(norm(Med,'fro')~=0) &(~isnan(norm(X,'fro'))) & (~isnan(norm(Med,'fro')))
%             norm(X,'fro')
%             norm(Med,'fro')
%             pause
            [d,Xnew,tr] = procrustes(Med,X);
            X=X*tr.T;
        end
        %Compute the direction Vi \in T_X(C) 
        V(i,:,:)=InverseExp(Med,X);
        d=norm(reshape(V(i,:,:),n,p),'fro');%acos(trace(Med*X'));
        dist=[dist d];
       % pause
        Vn(i,:,:)=V(i,:,:)./dist(i);
    end
   % keyboard
    %compute the average direction
    MuV=(1/(sum(1./dist)))*sum(Vn,1);
    MuV=squeeze(MuV(1,:,:));
    %keyboard
    Med=Exp(Med,eps2*MuV);
    E(j)=norm(MuV,'fro');
    j=j+1;
end

%   figure('PaperSize',[20.98 29.68],'Color',[1 1 1])
%   plot(E)

%keyboard