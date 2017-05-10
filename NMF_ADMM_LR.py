import numpy as np
import math


def D1(X, Y):
    ''' 
    Compute the Kullback Liebler Divergence of two Matrices
    
    Parameters
    ----------
    X: Matrix mxn
    Y: Matrix mxn
    
    Returns
    -------
    The KL divergence between the matrices D(X|Y)
    
    '''

    (m,n) = X.shape

    suma = 0.0
    for i in range(m):
        for j in range(n):
            suma += (X[i,j] * math.log(X[i,j] + 1e-12) - X[i,j] * math.log(Y[i,j] + 1e-12) - X[i,j] + Y[i,j])

    return suma


def nmf_WH(V, r, rho, max_iter):
    '''
    This NonNegative Matrix Factorization (NMF) based in the paper from D. L. Sun (see references)
    
    Parameters
    ----------
    V: Matrix mxn to be factorized
    r: Rank of the factorization
    rho: Parameter in the ADMM
    max_iter: Maximum number of iterations
    
    Returns
    -------
    W : Matrix in the factorization V = WH
    H : 
    
    References
    ----------
    D. L. Sun and C. Fevotte. Alternating direction method of multipliers 
    for non-negative matrix factorization with the beta-divergence, 
    IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), May 2014.
    http://statweb.stanford.edu/~dlsun/papers/nmf_admm.pdf
    http://www.calpoly.edu/~dsun09/admm.html
    
    Matlab Code
    -----------
    function [W, H, error_i] = nmf_admm(V, W, H, beta, rho, max_iter, fixed)
    % nmf_admm(V, W, H, rho, fixed)
    % 
    % Implements NMF algorithm described in:
    %   D.L. Sun and C. Fevotte, "Alternating direction method of multipliers 
    %      for non-negative matrix factorization with the beta divergence", ICASSP 2014.
    %
    % inputs
    %    V: matrix to factor 
    %    W, H: initializations for W and H
    %    beta: parameter of beta divergence 
    %          (only beta=0 (IS) and beta=1 (KL) are supported)
    %    rho: ADMM smothing parameter
    %    fixed: a vector containing the indices of the basis vectors in W to
    %           hold fixed (e.g., when W is known a priori)
    %
    % outputs
    %    W, H: factorization such that V \approx W*H
    
        %define the iteration
        error_i = zeros(1,max_iter);
        
        % determine dimensions
        [m,n] = size(V);
        [~,k] = size(W);
        
        % set defaults
        if nargin<5, rho=1; end
        if nargin<7, fixed=[]; end
        
        % get the vector of indices to update
        free = setdiff(1:k, fixed);
        
        % initializations for other variables
        X = W*H;
        Wplus = W;
        Hplus = H;
        alphaX = zeros(size(X));
        alphaW = zeros(size(W));
        alphaH = zeros(size(H));
        
        for iter=1:max_iter
            
            % update for H
            H = (W'*W + eye(k)) \ (W'*X + Hplus + 1/rho*(W'*alphaX - alphaH));
            
            % update for W
            P = H*H' + eye(k);
            Q = H*X' + Wplus' + 1/rho*(H*alphaX' - alphaW');
            W(:,free) = ( P(:,free) \ (Q - P(:,fixed)*W(:,fixed)') )';
            
            % update for X (this is the only step that depends on beta)
            X_ap = W*H;
            if beta==1
                b = rho*X_ap - alphaX - 1;
                X = (b + sqrt(b.^2 + 4*rho*V))/(2*rho);
            elseif beta==0
                A = alphaX/rho - X_ap;
                B = 1/(3*rho) - A.^2/9;
                C = - A.^3/27 + A/(6*rho) + V/(2*rho);
                D = B.^3 + C.^2;
    
                X(D>=0) = nthroot(C(D>=0)+sqrt(D(D>=0)),3) + ...
                    nthroot(C(D>=0)-sqrt(D(D>=0)),3) - ...
                    A(D>=0)/3;
    
                phi = acos(C(D<0) ./ ((-B(D<0)).^1.5));
                X(D<0) = 2*sqrt(-B(D<0)).*cos(phi/3) - A(D<0)/3;
            else
                error('The beta you specified is not currently supported.')
            end
    
            % update for H_+ and W_+
            Hplus = max(H + 1/rho*alphaH, 0);
            Wplus = max(W + 1/rho*alphaW, 0);
                
            % update for dual variables
            alphaX = alphaX + rho*(X - X_ap);
            alphaH = alphaH + rho*(H - Hplus);
            alphaW = alphaW + rho*(W - Wplus);
            
            %compute the error at iteration
            error_i(1,iter) = norm(V-Wplus*Hplus,'fro');
            
        end
            
        W(:,free) = Wplus(:,free);
        H = Hplus; 
            
    end
    
    
    Call the Matlab function
    ------------------------
    
    V = rand(100, 20);
    
    % initializations
    W0 = rand(100, 5);
    H0 = rand(5, 20);
    
    beta = 1; % set beta=1 for KL divergence, beta=0 for IS divergence
    rho = 1; % ADMM parameter
    
    [W, H] = nmf_admm(V, W0, H0, beta, rho)
    '''

    # Define dimension
    (m, n) = V.shape

    # Initialize W, H, X
    W = np.random.uniform(0.0,1.0, size=(m,r))
    H = np.random.uniform(0.0,1.0, size=(r,n))
    X = np.dot(W, H)

    # Initialize W_plus, H_plus
    W_plus = W[:,:]
    H_plus = H[:,:]

    # Initialize alpha_X, alpha_W, alpha_H
    alpha_X = np.zeros_like(X)
    alpha_W = np.zeros_like(W)
    alpha_H = np.zeros_like(H)

    for i in range(max_iter):

        # Compute H
        H_A1 = np.linalg.pinv( np.dot( np.transpose(W), W ) + np.eye(r) )
        H_B = np.dot( np.transpose(W), X ) + H_plus + (1.0/rho)*np.dot(np.transpose(W),alpha_X) - (1.0/rho)*alpha_H
        H = np.dot(H_A1, H_B)

        # Compute W
        W_A1 = np.linalg.pinv( np.dot(H,np.transpose(H)) + np.eye(r) )
        W_B = np.dot(H, np.transpose(X)) + np.transpose(W_plus) + (1.0/rho)*np.dot(H,np.transpose(alpha_X)) - (1.0/rho)*np.transpose(alpha_W)
        W = np.transpose( np.dot( W_A1  ,W_B ) )

        # Compute X
        b = rho*np.dot(W,H) - alpha_X - np.ones_like(X)
        X = (1.0/(2.0*rho))*b + (1.0/(2.0*rho))*( b**2+(4.0*rho)*V)**0.5

        # Compute W_plus
        W_plus = np.clip( W + (1.0/rho)*alpha_W, 0, np.inf )

        # Compute H_plus
        H_plus = np.clip( H + (1.0/rho)*alpha_H, 0, np.inf )

        # Compute alpha_W
        alpha_W = alpha_W + rho*W - rho*W_plus

        # Compute alpha_H
        alpha_H = alpha_H + rho*H - rho*H_plus

        # Compute alpha_X
        alpha_X = alpha_X + rho*X - rho*np.dot(W,H)

    return W_plus, H_plus


if __name__ == '__main__':

    # Generar ejemplo
    m = 100
    n = 100
    r = 5

    # Matriz sin ruido
    W0 = np.random.uniform(0.0, 5.0, size=(m, r))
    H0 = np.random.uniform(0.0, 5.0, size=(r, n))

    V0 = np.dot(W0, H0)

    max_V0 = np.max(V0)
    V0_norm = (1.0 / max_V0) * V0

    # Resolver
    W, H = nmf_WH(V0_norm, r=5, rho=0.1, max_iter=2000)
    print D1(V0_norm, np.dot(W,H))

