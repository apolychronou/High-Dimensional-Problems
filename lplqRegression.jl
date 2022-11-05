using DelimitedFiles;
using Statistics;
using LinearAlgebra;
using StatsBase;
using RegularizationTools;
using ProgressMeter;
using SparseArrays;


"""
This function solves the minimizes
1/p*||X*beta-y||_p^p+1/q*||y||_q^q (1)
with the Majorization-Minimization method (adaptive quadratic majorant).

beta = lplqRegression(X,y,p,q,mu,options);

INPUT:
X:       mxn system matrix in (1) (can be also an object, provided it can
         perform products with vector and matrices and transposition);\n
y:       mx1 right-hand side vector in (1);\n
p:       norm for the data fitting term in (1), 0<p<=2;\n
q:       norm for the regularization term in (1), 0<q<=2;\n
mu:      regularization paramter in (1), mu>0;\n
options: optional input, structure that can contain the following fields:\n
         - epsilon:        smoothing parameter for the MM method
                           (default=1e-4);\n
         - tollerance:     tollerance for the stopping criterion of the
                           MM method (default=1e-4);\n
         - max_iterations: maximum number of iterations of the MM method
                           (default=500);\n
         - waitbar:        if 'off' than the waitbar is not shown,
                           otherwise is displayed (default='off').\n

Alessandro Buccini, 27 Jan 2020

"""
function lplqRegression(X,y,p,q,mu,options=[])

    if p<=0 || p>2
        error("Invalid value for p");
    elseif q<=0 || q>2
        error("Invalid value for q");
    elseif mu<0
        error("Invalid value for mu");
    end

    if ~hasproperty(options,:epsilon)
        epsilon=1e-4;
    else
        epsilon=options.epsilon;
    end
    if ~hasproperty(options,:tolerance)
        tol=1e-4;
    else
        tol=options.tolerance;
    end
    if ~hasproperty(options,:max_iterations)
        iter=500;
    else
        iter=options.max_iterations;
    end
    if ~hasproperty(options,:waitbar)
        wb="off";
    else
        wb=options.waitbar;
    end

    # initial guess
    beta=X'*y;

    #creating initial space
    l=min(length(beta),5);
    V=lanc_b(X,y,l);
    # creating projected matrix
    XV=X*V;
    LV=V;

    # progress
    if ~ (wb=="off")
        quiet=false;
    else
        quiet=true;
    end
    prog = quiet ? nothing : Progress(iter)

    # begin MM iterations
    for k in 1:iter
        #Store previous iteration for stopping criterion
        # beta_old=copy(beta);
        beta_old=(beta);

        # Compute weights for approximating p/q norms with the 2 norm
        v=X*beta-y;
        # u=copy(beta);
        u=(beta);
        # wf=((v.^2 .+epsilon^2).^(p/2-1));
        wf=copy((v.^2 .+epsilon^2).^(p/2-1));
        Wf=spdiagm(0=>convert(Vector,wf[:,1]));
        # wr=((u.^2 .+epsilon^2).^(q/2-1));
        wr=copy((u.^2 .+epsilon^2).^(q/2-1));

        Wr=spdiagm(0=>convert(Vector,wr[:,1]));

        # compute QR factorization
        n=minimum(size(Wf.^(1/2)*XV));
        F1=qr(Wf.^(1/2)*XV);
        QA=F1.Q;
        RA=F1.R;


        F2=qr(Wr.^(1/2)*V);
        RL=F2.R;

        z=[RA;sqrt(mu)*RL]\[QA[:,1:n]'*(Wf*y) ; zeros(size(RL,2),1)];
        beta=V*z;


        if norm(beta-beta_old)/norm(beta_old)<tol && k>1
            break
        end

        if k+l<length(beta)
            #compute residual
            r=X'*(Wf*(XV*z-y))+mu*Wr*(LV*z);
            r=r-V*(V'*r);
            r=r-V*(V'*r);

            # Enlarge space
            vn=(r/norm(r));
            V=([V vn]);
            XV=([XV X*vn]);
            LV=[LV vn];
        else
            # XV=(X);
            XV=copy(X);
            V=I
            LV=copy(V);
            # LV=(V);

        end
        quiet ? nothing : sleep(0);
        quiet ? nothing : next!(prog);

    end
    return beta;
end



function lanc_b(A,p,k)
    #initialization
    m,n=size(A);
    U=zeros(m,k);
    V=zeros(n,k);

    #prepare for Lanczos iteration
    v=zeros(n);
    beta=norm(p);
    u=copy(p/beta);
    # u=(p/beta);
    U[:,1]=u;

    #perform Lanczos bidiagonalization with reorthogonalization
    for i in 1:k
        r=A'*u-beta*v;
        for j in 1:i-1
            d=(V[:,j]'*r)[1,1]
            r-=d*V[:,j];
        end
        alpha=norm(r);
        v=(r/alpha);
        V[:,i]=v;
        if i==k
            break
        end
        p=A*v-alpha*u;
        for j=1:i
            z=((U[:,j]'*p)[1,1]);
            p=p-z*U[:,j];
        end
        beta=norm(p);
        u=(p/beta);

        U[:,i+1]=u;

    end
    return V;
end
