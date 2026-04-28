function A=ComputeGeodesicSNew(Source, Target, leng)


X=Source;
Y=Target;
% X=CenteredScaled(Source);
% Y=CenteredScaled(Target);

% X=Source;
% Y=Target;
if (norm(X,'fro')~=0)&(norm(Y,'fro')~=0)&(~isnan(norm(X,'fro'))) & (~isnan(norm(Y,'fro')))
       [d,YT,tr] = procrustes(X,Y,'reflection',0);
else
    X=ones(25,3); Y=ones(25,3);
    tr.T=eye(3);
end


%Procrustes Analysis

YT=Y;%*tr.T;
% compute geodesic
theta=acos(trace(X*YT'));
%A=X;
for t=1:leng
 %   clf
    A(:,t,:)=CenteredScaled((1./sin(theta)).*(sin(1-(t/leng)*theta)*X+sin((((t/leng))*theta)*YT)));  
end


%     figure('PaperSize',[20.98 29.68],'Color',[1 1 1])
%     DrawSkeletonSequenceAction3D(A,3,'b','r') 
%     axis off
%     box on
%     grid on
%     set(gca, 'XTick', []);
%     set(gca, 'YTick', []);
%     set(gca, 'ZTick', []);
%     axis on
%     grid on
%     
%     figure('PaperSize',[20.98 29.68],'Color',[1 1 1])
%     DrawSkeletonframeAction3D(squeeze(A(:,1,:)),'b','r');
%     view(175,-86)
%     axis off
% %     
%     figure('PaperSize',[20.98 29.68],'Color',[1 1 1])
%     DrawSkeletonframeAction3D(squeeze(A(:,7,:)),'b','r');
%     view(175,-86)
%     axis off
%     
%     figure('PaperSize',[20.98 29.68],'Color',[1 1 1])
%     DrawSkeletonframeAction3D(squeeze(A(:,14,:)),'b','r');
%     view(175,-86)
%     axis off
%     
%     figure('PaperSize',[20.98 29.68],'Color',[1 1 1])
%     DrawSkeletonframeAction3D(squeeze(A(:,22,:)),'b','r');
%     view(175,-86)
%     axis off
    
    
    %Compute V(X->Y) the parallel transport of V to Y
    %Compute the tangent vector on 
    
%     V=(theta/sin(theta))*(YT-cos(theta)*X);
%     VY=(theta/sin(theta))*(X-cos(theta)*YT);
%     %magnitude of V on each joint of the skt
%     for p=1:20
%         MV(p)=norm(V(p,:),1);
%     end
% 
%     %The parallel transport of V to Y
%     VXY=V-((2*trace(V*YT'))/norm(X+YT,'fro'))*(X+YT);
%     acos(trace(VXY*VY'))
end