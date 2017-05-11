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


def remove_cols(A, indices ):
    ''' 
    Resize matrix A by removing the columns in indices

    Parameters
    ----------
    A: Matrix mxn to be resized
    indices: Columns to be removed from matrix A

    Returns
    -------
    The resized matrix A
    '''

    (m_old, n_old) = A.shape
    n_new = len(indices)

    A_new = np.empty((m_old, n_new))

    for j in range( n_new ):
        A_new[:,j] = A[ :, indices[j] ]

    return A_new


def remove_rows( A, indices ):
    ''' 
    Resize matrix A by removing the rows in indices

    Parameters
    ----------
    A: Matrix mxn to be resized
    indices: Rows to be removed from matrix A

    Returns
    -------
    The resized matrix A
    '''

    (m_old, n_old) = A.shape
    m_new = len(indices)

    A_new = np.empty((m_new,n_old))

    for i in range(m_new ):
        A_new[i,:] = A[indices[i],:]

    return A_new


def remove_index(x, indices):
    ''' 
    Resize vector x by removing the elements in indices

    Parameters
    ----------
    x: Vector to be resized
    indices: Elements to be removed from vector x

    Returns
    -------
    The resized vector
    '''

    n_new = len(indices)

    x_new = np.empty( n_new )

    for i in range( n_new ):
        x_new[i] = x[indices[i]]

    return x_new


def nmf_WH(V, r, rho, max_iter):
    ''' 
    This NonNegative Matrix Factorization (NMF) based in the paper from D. L. Sun (see references):
    
    min D(V|X)
    s.t.
        X = WH
        W = W_+
        H = H_+
        W_+, H_+ >= 0
    
    Parameters
    ----------
    V: Matrix mxn to be factorized
    r: Rank of the factorization
    rho: Parameter in the ADMM
    max_iter: Maximum number of iterations
    
    Returns
    -------
    W: Matrix in the factorization V = WH
    H: Matrix in the factorization V = WH
    
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


def nmf_WDH(V, r0, mu, rho, ranges):
    ''' 
    This NonNegative Matrix Factorization (NMF) estimates the optimal (minimum) rank for the factorization by solving:

    min D(V|X) + \mu tr(D_+)
    s.t.
        X = WDH
        W = W_+
        D = D_+
        H = H_+
        W_+, D_+, H_+ >= 0

    with D(V|X) = Kullback-Leibler Divergence

    Parameters
    ----------
    V: Matrix mxn to be factorized
    r0: Initial rank of the factorization
    mu: Penalization parameter
    rho: Parameter in the ADMM
    ranges: Iterations to be performed. Last number represents the max iters in the algoritm. The intermediate numbers
            represent the iters where the rank of the factorization will be updated (removing the zero elements)

    Returns
    -------
    W: Matrix in the factorization V = WDH
    D: Diagonal matrix in the factorization V = WDH
    H: Matrix in the factorization V = WDH

    References
    ----------
    Personal work!
    '''

    (m, n) = V.shape
    rho_1 = 1.0 / rho

    # Initialize W,d,H, X
    W = np.random.uniform(0.0, 1.0, size=(m,r0))
    d = np.ones(r0)
    H = np.random.uniform(0.0, 1.0, size=(r0,n))
    X = np.dot(W, np.dot(np.diag(d),H))

    # Initialize W_plus, d_plus, H_plus
    W_plus = np.copy( W )
    d_plus = np.copy( d )
    H_plus = np.copy( H )

    # Initialize alpha_X, alpha_W, alpha_D, alpha_H
    alpha_X = np.zeros( (m,n) )
    alpha_W = np.zeros( (m,r0) )
    alpha_D = np.zeros( (r0) )
    alpha_H = np.zeros( (r0,n) )

    # Initialize P_w
    P_w = np.ones( (r0) )

    # Initialize Irr
    Irr = np.eye(r0)

    last = 0
    for j in range(len(ranges)):

        for i in range(ranges[j] - last):

            # Compute H
            WD   = np.dot(W, np.diag(d))
            H_A1 = np.linalg.pinv(np.dot(np.transpose(WD), WD) + Irr)
            H_B  = np.dot(np.transpose(WD), X + rho_1*alpha_X) + H_plus - rho_1*alpha_H
            H    = np.dot(H_A1, H_B)

            # Compute W
            DH   = np.dot(np.diag(d), H)
            W_A1 = np.linalg.pinv(np.dot(DH, np.transpose(DH)) + Irr)
            W_B  = np.dot(X + rho_1*alpha_X, np.transpose(DH)) + W_plus - rho_1*alpha_W
            W    = np.dot(W_B, W_A1)

            # Compute d
            d_A1 = np.linalg.pinv(np.dot(H, np.transpose(H)) * np.dot(np.transpose(W), W) + Irr)
            d_B  = np.diag(np.dot(H, np.dot(np.transpose(X + rho_1*alpha_X), W))) + d_plus - rho_1*alpha_D
            d    = np.dot(d_A1, d_B)

            # Compute X
            b = rho*np.dot(W, np.dot(np.diag(d), H)) - alpha_X - np.ones_like(X)
            X = (1.0/(2.0*rho))*b + (1.0/(2.0*rho))*(b**2+(4.0*rho)*V)**0.5

            # Compute W_plus
            W_plus = np.clip(W + rho_1*alpha_W, 0, np.inf)

            # Compute H_plus
            H_plus = np.clip(H + rho_1*alpha_H, 0, np.inf)

            # Compute D_plus
            d_plus = np.clip(d + rho_1*alpha_D - rho_1*mu[j], 0, np.inf)

            # Compute alpha_X
            alpha_X = alpha_X + rho*X - rho*np.dot(W, np.dot(np.diag(d), H))

            # Compute alpha_W
            alpha_W = alpha_W + rho*W - rho*W_plus

            # Compute alpha_H
            alpha_H = alpha_H + rho*H - rho*H_plus

            # Compute alpha_D
            alpha_D = alpha_D + rho*d - rho*d_plus

        # ----------------- RESIZE ----------------- #
        active_set = []
        for k in range(d_plus.shape[0]):
            if d_plus[k] > 0.0:
                active_set.append(k)

        # Reshape W,d,H, X
        W = remove_cols(W, active_set)
        H = remove_rows(H, active_set)
        d = remove_index(d, active_set)
        X = np.dot(W, np.dot(np.diag(d), H))

        # Reshape W_plus, H_plus, d_plus
        W_plus = remove_cols(W_plus, active_set)
        H_plus = remove_rows(H_plus, active_set)
        d_plus = remove_index(d_plus, active_set)

        # Reshape alafa_X, alpha_W, alpha_H, alpha_D
        alpha_X = np.zeros((m, n))
        alpha_W = np.zeros((m,len(active_set)))
        alpha_H = np.zeros((len(active_set),n))
        alpha_D = np.zeros((len(active_set)))

        # Reshape Irr
        Irr = np.eye(len(active_set))
        # ----------------- RESIZE ----------------- #

        last = ranges[j]

    return W_plus, d_plus, H_plus




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

    # Resolver V = WH
    W, H = nmf_WH(V0_norm, r=5, rho=0.1, max_iter=2000)
    print D1(V0_norm, np.dot(W,H))

    # Resolver V = WDH
    iters = [50, 100, 200, 500, 1000, 2000]
    mu = [1.0, 0.5, .5, .5, .5, .5]
    rho = 0.1
    W, d, H = nmf_WDH(V0_norm, r0=100, mu=mu, rho=rho, ranges=iters)
    print d.shape, D1(V0_norm, np.dot(W, np.dot(np.diag(d), H)))


