function V=InverseExp(X,Y) %exp-1_X(Y)
if(norm(X-Y,'fro')<0.00001)
    theta=0.0001;
    disp('theta too small')
else
    theta=acos(InnerProd_Q(X,Y));
end
%theta
V=(theta/sin(theta))*(Y-cos(theta)*X);
