%%%%% This Program illustrates the rate-invariant analysis of trajectories
%%%%% on Kendall's shape space for action recognition as described in 
%%%%% Boulbaba Ben Amor, Jingyong Su, Anuj Srivastava: "Action Recognition Using Rate-Invariant 
%%%%% Analysis of Skeletal Shape Trajectories". IEEE Trans. Pattern Anal. Mach. Intell. 38(1): 1-13 (2016)

clear all
close all
clc

%%%%%%%%%%%%%%%%% READ DATA

% Good examples act = 11, sub1 = 7, sub2 = 9
%   [8 9 6]
%   [11 1 10]

% act = ceil(rand*20); 
% sub1 = ceil(rand*10); 
% sub2 = sub1;
% while sub2 == sub1
%     sub2 = ceil(rand*10);
% end

act = 11;
sub1 = 7;
sub2 = 9;

[act sub1 sub2]
Seq = sprintf('MSRAction3DSkeletonReal3D/a%02d_s%02d_e01_skeleton3D.txt',act,sub1);
Seq2 = sprintf('MSRAction3DSkeletonReal3D/a%02d_s%02d_e01_skeleton3D.txt',act,sub2);

B=ReadSkeletonAction3D(Seq);

%%

%Seq=strcat('MSRAction3DSkeletonReal3D/a11_s01_e01','_skeleton3D.txt');
%Seq2=strcat('MSRAction3DSkeletonReal3D/a11_s10_e02','_skeleton3D.txt');

%%%%%%%%%%%%%%%%%%% DENOSING
w=4; %sliding window size = 2w+1
TT=Denoising(ReadSkeletonAction3D(Seq),4);
TT2=Denoising(ReadSkeletonAction3D(Seq2),4);

%%%%%%%%%%%%%%%%%%% RESAMPLING
s=0:0.01:1; %resampling step
lenS = length(s)-1;
T= ResamplingFilter(TT,s); 
T2= ResamplingFilter(TT2,s);
Ref_c = squeeze(T(:,1,:));

%%%%%%%%%%%%%%%%%%% ALIGNMENT
[TAlig, dist, dist_align,gam]=Align2SequencesDist(T,T2,lenS, Ref_c);
%dist -- distance before alignment
%dist_align -- distance after alignment

%%%%%%%%%%%%%%%%%%% DISPLAY RESULTS
%Before alignment
figure('Name','Before temporal alignment','NumberTitle','off','PaperSize',[20.98 29.68],'Color',[1 1 1])
DrawSkeletonSequenceAction3DAnimation2(T,T2,2,'b','r','g','k');%TAlig
pause(0.1);
%After alignemt
figure('Name','After temporal alignment','NumberTitle','off','PaperSize',[20.98 29.68],'Color',[1 1 1])
DrawSkeletonSequenceAction3DAnimation2(TAlig,T2,2,'k','r','g','k');%
pause(0.1);
%Optimal re-parametrization
figure('Name','Optimal re-parametrization','PaperSize',[20.98 29.68],'Color',[1 1 1])
plot((1:lenS)/lenS,gam,'r','LineWidth',2); %[0:1:(100-1)]/(100-1)
legend('Optimal re-parametrization \gamma^*','Location','northwest','Orientation','horizontal')
title(['Distance before \color{red}{',num2str(sum(dist)),'}\color{black}{ Distance after }\color{blue}',num2str(sum(dist_align))])
grid on
