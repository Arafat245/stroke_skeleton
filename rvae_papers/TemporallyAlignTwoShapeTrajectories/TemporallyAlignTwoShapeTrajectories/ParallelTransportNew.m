function V_par = ParallelTransportNew(X,V0,Y)
X0=CenteredScaled(X);
Y0=CenteredScaled(Y);
Y0_new=Y0;
if (norm(X0,'fro')~=0)&(norm(Y0,'fro')~=0)&(~isnan(norm(X0,'fro'))) & (~isnan(norm(Y0,'fro')))
        [d,Y0_new,B] = procrustes(X0,Y0);
% else
%     X0=ones(31,3); Y0=ones(31,3);
%     B.T=eye(3);
end

% %%%%%%%%%%%%%%%%%%%% OLD CODE
% tic
% V_par=V0-((2*trace(V0*Y0_new')/norm(X0+Y0_new,'fro'))*(X0+Y0_new));
% toc

%%%%%%%%%%%%%%%%%%%%% NEW CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic
leng=10;
X_s=ComputeGeodesicSNew(X0,Y0_new,leng);
X_point=DeriveGeodesic(X_s);

%Define delta
delta=1/leng;
V_new=V0';
for i=1:leng-1
    S=squeeze(X_s(:,i,:))'*squeeze(X_s(:,i,:));
    B=(squeeze(X_point(:,i,:))'*V_new')-(V_new*squeeze(X_point(:,i,:)));
    %cheked S is symmetric and B skew-sym.
    A=Compute_A(S,B);
    
    Y=V_new-(trace(squeeze(X_point(:,i,:))'*V_new')*squeeze(X_s(:,i,:))')*delta+A*squeeze(X_point(:,i,:))'*delta;
    
    S=squeeze(X_s(:,i+1,:))'*squeeze(X_s(:,i+1,:));
    B=Y*squeeze(X_s(:,i+1,:))-squeeze(X_s(:,i+1,:))'*Y';
    %B=V_new*squeeze(X_s(:,i,:))-squeeze(X_s(:,i,:))'*V_new';
    NewC=Compute_A(S,B);
    
    %keyboard
    YY0=Y-NewC*squeeze(X_point(:,i,:))';
    V_new=YY0*(norm(Y,'fro')/norm(YY0,'fro'));
end
V_par=V_new';
%toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

%norm(V_par'-V_new,'fro')
