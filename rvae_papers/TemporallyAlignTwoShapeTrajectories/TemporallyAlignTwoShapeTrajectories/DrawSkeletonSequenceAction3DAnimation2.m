function h=DrawSkeletonSequenceAction3DAnimation2(skeleton_bis,skeleton_bis2,k,Ec,Lc,Ec2,Lc2)
    %skeleton_bis contains the skeleton
   
    %k is the step
    
    a=[20     1     2     1     8    10     2     9    11     3     4     7     7     5     6    14    15    16    17];
    b=[3     3     3     8    10    12     9    11    13     4     7     5     6    14    15    16    17    18    19];
    [n,m,t]=size(skeleton_bis);   %m - the frame #, n - the joint number, k - the 3D coordinates 
        
    for i=1:k:m
        clf
        plot3(skeleton_bis(1:n,i,1),skeleton_bis(1:n,i,2),skeleton_bis(1:n,i,3) ,'.','MarkerSize',30,'MarkerFaceColor',Lc,'MarkerEdgeColor',Lc);
        hold on
        plot3(skeleton_bis2(1:n,i,1),skeleton_bis2(1:n,i,2),skeleton_bis2(1:n,i,3) ,'.','MarkerSize',30,'MarkerFaceColor',Lc2,'MarkerEdgeColor',Lc2);

        for j=1:length(a)
            line([skeleton_bis(a(j),i,1) skeleton_bis(b(j),i,1)],[skeleton_bis(a(j),i,2) skeleton_bis(b(j),i,2)],[skeleton_bis(a(j),i,3) skeleton_bis(b(j),i ,3)],'Color',Ec,'LineWidth',3);
            line([skeleton_bis2(a(j),i,1) skeleton_bis2(b(j),i,1)],[skeleton_bis2(a(j),i,2) skeleton_bis2(b(j),i,2)],[skeleton_bis2(a(j),i,3) skeleton_bis2(b(j),i ,3)],'Color',Ec2,'LineWidth',3);

            axis equal
            axis off
            hold on,
          %  view([-182 -82])
            view([-182 -70])
        end
         pause(0.001)
    end

end

