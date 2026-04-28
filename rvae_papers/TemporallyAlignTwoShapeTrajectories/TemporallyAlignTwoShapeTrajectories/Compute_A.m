%Problem1
%input S:symmetric define positive, B: skew-sym s.t.
%output A | AS+SA=B 
function A=Compute_A(S,B)
    [R,MU,R2]=svd(S);
    B_tilda=R'*B*R;
    %size(B_tilda)
    [m,n]=size(S);
    D=diag(MU);
%     keyboard
    for i=1:m
        for j=1:n
            A_tilda(i,j)=B_tilda(i,j)./(D(i)+D(j));
        end
    end
    A=R*A_tilda*R';
%      B
%      A*S+S*A
%     keyboard
end