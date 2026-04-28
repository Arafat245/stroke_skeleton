function X_point=DeriveGeodesic(X_s)
    X_point=diff(X_s,1,2);

%     leng=size(X_s,2);
%     X=squeeze(X_s(:,1,:));
%     Y=squeeze(X_s(:,leng,:));
%     X=CenteredScaled(X);
%     Y=CenteredScaled(Y);
% 
%     %Procrustes Analysis
%     [d,YT,tr] = procrustes(X,Y);
%     %YT=Y*tr.T;
%     % compute geodesic
%     theta=acos(trace(X*YT'));
%     %A=X;
%     for t=1:leng
%      %   clf
%         X_point(:,t,:)=-(theta/sin(theta))*(cos(1-(t/leng)*theta)*X-cos((((t/leng))*theta)*YT));    
%     end
%    keyboard
end