function Y=Exp(X,V) %exp-1_X(V)
theta=sqrt(InnerProd_Q(V,V));
Y=cos(theta)*X+(sin(theta)/theta)*V;
