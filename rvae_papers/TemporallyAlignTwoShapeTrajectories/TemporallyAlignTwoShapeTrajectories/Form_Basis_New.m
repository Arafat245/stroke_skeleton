% Construct an orthonormal basis set given a reference point C

function F_OrtNorm=Form_Basis_New(Ref_c)

n=size(Ref_c,1);

%Sub-matrix H
for j=1:n-1
    h_j=-1/sqrt(j*(j+1));
    H(j,:)=[repmat(h_j,1,j),-j*h_j,zeros(1,n-j-1)];
end

C=H*Ref_c; %Space C_0
C=C/sqrt(InnerProd_Q(C,C));%Pre-shape space C

% Etilde? R^{n x 3}
for i=1:3*n
    Etmp=zeros(n,3);
    Etmp(i)=1;
    Etilde{i}=H*Etmp;
end

%{A1,A2,A3}:an orthonormal basis for 3 x 3 skew symmetric matrices
A{1}=1/sqrt(2)*[0 1 0;-1 0 0;0 0 0];
A{2}=1/sqrt(2)*[0 0 1;0 0 0;-1 0 0];
A{3}=1/sqrt(2)*[0 0 0;0 0 1;0 -1 0];

%Orthogonalize CA matrices
for j=1:3
    tt = C*A{j};
    CMat(:,j) = reshape(tt,(n-1)*3,1);
end
CO = orth(CMat);
Cm{1} = reshape(CO(:,1),(n-1),3);
Cm{2} = reshape(CO(:,2),(n-1),3);
Cm{3} = reshape(CO(:,3),(n-1),3);

%Construct 3*n-7 orthonormal basis: each basis is a column of length 3(n-1)
for i=1:3*n
     Fn{i}=Etilde{i}-InnerProd_Q(Etilde{i},C)*C;
     for j=1:3
         Fn{i}=Fn{i}-InnerProd_Q(Fn{i},Cm{j})*Cm{j}; 
     end
     FnMat(:,i) = reshape(Fn{i},(n-1)*3,1);
end

F_OrtNorm = orth(FnMat);


