function Filtered1=Denoising(skeleton1,w)
[n1,m1,k1]=size(skeleton1);
n=floor(m1);
Step=1;
for i=1:n % each skeleton --> C
    X=squeeze(skeleton1(1:n1,Step*i,1:3));
    alpha_C(i,:,:)=CenteredScaled(X);
    alpha_C1(:,i,:)=CenteredScaled(X);
end
% keyboard
%Apply the median filter
Filtered=alpha_C;Filtered1=alpha_C1;
[n,m,p]=size(alpha_C);
for i=w+1:n-w-1
    Filtered(i,:,:)=Median(alpha_C(i-w:i+w,:,:));
    Filtered1(:,i,:)=Median(alpha_C(i-w:i+w,:,:));
end



