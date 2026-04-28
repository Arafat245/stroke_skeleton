function B=ReadSkeletonAction3D(filename)
B=[];
%file=sprintf(filename);
fp=fopen(filename);
if (fp>0)
  A=fscanf(fp,'%f');
  B=[B; A];
  fclose(fp);
end
l=size(B,1)/4;
B=reshape(B,4,l);
B=B';
B=reshape(B,20,l/20,4);