using DelimitedFiles;
using Statistics;
using LinearAlgebra;
using Random;
using Gadfly;
using Printf;
using StatsBase;
using RegularizationTools;
using Lazy;
using MultivariateStats;
using DataFrames;
import Base.Threads.@spawn
using ProgressMeter;
using Distributions;

"""
picard(X,y) \n
Computes the plot so as to check the picard condition
"""
function picard(X,b,file)
    y=X*b
    ŷ=y+randn(size(y,1));
    U,S,~=svd(X);
    c=U'*y;
    ĉ=U'*ŷ;
    c = abs.(c) ./S
    ĉ = abs.(ĉ) ./S

    pic=plot(Scale.color_discrete,Guide.title("Picard Condition2"));
    push!(pic,layer(x=collect(1:length(S)),y=log10.(c),color=["without noise"],Geom.point, Geom.line));
    push!(pic,layer(x=collect(1:length(S)),y=log10.(ĉ),color=["with noise"],Geom.point, Geom.line));
    draw(SVG("picard2_"*file[1:end-4]*".svg", 7inch, 7inch), pic);

    c=broadcast(abs,U'ŷ);

    pic=plot(Scale.color_discrete,Guide.title("Picard Condition1"));
    push!(pic,layer(x=collect(1:length(S)),y=log10.(c),color=["u^tb"],Geom.point, Geom.line));
    push!(pic,layer(x=collect(1:length(S)),y=log10.(S),color=["si"],Geom.point, Geom.line));
    draw(SVG("picard1_"*file[1:end-4]*".svg", 7inch, 7inch), pic);




    return;
end

function picard_pca(X,b,file)
    y=X*b
    ŷ=y+randn(size(y,1));
    U,S,~=svd(X);
    c=U'*y;
    ĉ=U'*ŷ;
    c = abs.(c) ./S
    ĉ = abs.(ĉ) ./S

    pic=plot(Scale.color_discrete,Guide.title("Picard Condition2-PCA"));
    push!(pic,layer(x=collect(1:length(S)),y=log10.(c),color=["without noise"],Geom.point, Geom.line));
    push!(pic,layer(x=collect(1:length(S)),y=log10.(ĉ),color=["with noise"],Geom.point, Geom.line));
    draw(SVG("picard2_pca_"*file[1:end-4]*".svg", 7inch, 7inch), pic);

    c=broadcast(abs,U'y);

    pic=plot(Scale.color_discrete,Guide.title("Picard Condition1-PCA"));
    push!(pic,layer(x=collect(1:length(S)),y=log10.(c),color=["u^tb"],Geom.point, Geom.line));
    push!(pic,layer(x=collect(1:length(S)),y=log10.(S),color=["si"],Geom.point, Geom.line));
    draw(SVG("picard1_pca_"*file[1:end-4]*".svg", 7inch, 7inch), pic);




    return;
end
"""
η=eff₋cond(X,y,n=2)\n
Computes the effective condition number of X,y with default norm=2
"""
function eff_cond(X,y,n=2)
    X⁺=pinv(X);
    η=norm(X⁺,n)*norm(y,n)/norm(X⁺*y,n);
    return η;
end


"""
Compute mean square error of vectors x,y
"""
function mse(x,y)
    mse=mean((x-y).^2);
    return mse
end






struct options
    epsilon
    tolerance
    max_iterations
    waitbar
end



"""
Data=b₋iterations(X,b₋range,b₋iter,e₋iter,method)\n
Compare regressions methods OLS, Tikhonov, LpLq \n
Using #b₋iter randn b, each for #e₋iter noise randn \n
b₋range can be implemented to change the range of b in each iteration \n
"""
function b_iterations(Xin,b_range=1,b_iter=3,e_iter=100;method="lin",pca="false")
    Random.seed!(354615);
    minp,ninp=size(Xin);
    stats_ran_b=zeros(b_iter*b_range,e_iter);
    r=1;
    local b̂;
    if pca=="true"
        M=fit(PCA,Xin'); #;pratio=1.0
        X=Matrix(MultivariateStats.transform(M,Xin')');
    else
        X=Xin;
    end
    m,n=size(X);
    if method=="lin"
        X⁺=pinv(X);
    elseif method=="tikh"
        Ψ = setupRegularizationProblem(X, 0);
    elseif method=="lplq"
            p=2;
            q=0.1;
            op=options(1e-4,1e-4,50,"off");
    end
    for k in 1:b_range
        # N=Normal(0,r);
        for j in 1:b_iter;
            bin=randn(ninp);
            if pca=="true"
                b=MultivariateStats.transform(M,bin);
            else
                b=bin;
            end
            y=X*b;

            for i in 1:e_iter;
                ŷ=y+rand(m);
                if method=="lin"
                    b̂=X⁺*ŷ;
                elseif method=="tikh"
                    solution = solve(Ψ, ŷ)
                    b̂ = solution.x
                elseif method=="lplq"
                    local minnorm=floatmax();
                    for mu=10 .^range(-7,stop=-2,length=100)
                        br=lplqRegression(X,ŷ,p,q,mu,op);
                        if mse(br,b)<minnorm
                            minnorm=mse(br,b);
                            b̂=copy(br);
                        end
                    end
                end
                if pca=="true"
                    b̂=reconstruct(M,b̂);
                end
                stats_ran_b[(k-1)*b_iter+j,i]=mse(b̂,bin);
            end
        end
        # r=3*r;
    end

    return convert(DataFrame,stats_ran_b');
end
