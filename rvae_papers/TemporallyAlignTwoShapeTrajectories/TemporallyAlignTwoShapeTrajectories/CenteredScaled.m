function X0=CenteredScaled(X)
    n=size(X,1);
    muX = mean(X,1);
    X0 = X - repmat(muX, n, 1);

   % the "centered" Frobenius norm
    normX = sqrt(trace(X0*X0'));

    % scale to equal (unit) norm
    X0 = X0 / normX;
