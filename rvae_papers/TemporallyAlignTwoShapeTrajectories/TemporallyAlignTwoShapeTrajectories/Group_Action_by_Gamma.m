function pn = Group_Action_by_Gamma(p,gamma)

[n,T] = size(p);
pn=spline(linspace(0,1,T) , p ,gamma);
% q_composed_gamma = spline(linspace(0,1,T) , q ,gamma);
% sqrt_gamma_t = repmat(sqrt(gamma_t),n,1);
% qn = q_composed_gamma .*sqrt_gamma_t ;
% 
% qn=qn';
