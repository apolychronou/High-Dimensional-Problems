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
using Distributions;

include("lplqRegression.jl");
include("project_functions.jl");




    Random.seed!(1234);
set_default_plot_size(500px,
                      500px)


path="\\Users\\Thanos\\Documents\\Efarmosmenh Grammikh Algebra\\Project\\design matrices\\DM_same_variance_correlation\\r_0.9_sigma2_0.25/";

dir=readdir(path);
#loop for files
for fileLoop in 2:2:10


local file=dir[fileLoop];
println(file);
cd(path*file[1:end-4]);
#redirect stdout to file
open("stats_"*file,"a") do io
   redirect_stdout(io) do

X = readdlm(path*file,'\t')
XX = hcat(ones(size(X,1),1),X);
cr=cor(X);#correlation matrix
@printf("Condition Number of %s is: %f\n",file,cond(X));
draw(SVG("cor_"*file[1:end-4]*".svg", 7inch, 7inch), spy(cr));
M=fit(PCA,XX');#pca
println(M);
X_pca=Matrix(MultivariateStats.transform(M,XX')');#pca transformation
cr2=cor(X_pca);
draw(SVG("cor_pca_"*file[1:end-4]*".svg", 7inch, 7inch), spy(cr2));

#print picard plots and effective condition
ra=randn(size(XX,2))
noise=randn(size(XX,1))
ra_pca=MultivariateStats.transform(M,ra);
picard(XX,ra,file);
picard_pca(X_pca,ra_pca,file);
η=eff_cond(XX,XX*ra+noise);
print("Original effective condition number: $η\n");
η=eff_cond(X_pca,X_pca*ra_pca+noise);
print("PCA effective condition number: $η\n");

#run test function b₋iter for each method
e_iter=100;

@time local stats1=b_iterations(XX,1,3,e_iter;method="lin",pca="false");
@time local stats2=b_iterations(XX,1,3,e_iter;method="lin",pca="true");

@time local stats3=b_iterations(XX,1,3,e_iter;method="tikh",pca="false");
@time local stats4=b_iterations(XX,1,3,e_iter;method="tikh",pca="true");

@time local stats5=b_iterations(XX,1,3,e_iter;method="lplq",pca="false");
@time local stats6=b_iterations(XX,1,3,e_iter;method="lplq",pca="true");


#print mse in file and do mse plot
local mean_stats=zeros(6,3);
local pic=plot(Scale.color_discrete,Guide.title("MSE"));
local i,j;
local i=1;
local j=1;
for x in eachcol(stats1)
    y=mean(x);
    # print(y);
    # print("  ");
    mean_stats[i,j]=y;
        j=j+1;
end
push!(pic,layer(x=collect(1:3),y=mean_stats[i,:],color=["lin"]));
# print("\n\n")
i=2;
j=1;
for x in eachcol(stats2)
    y=mean(x);
    # print(y);
    # print("  ")
    mean_stats[i,j]=y;
        j=j+1;
end
push!(pic,layer(x=collect(1:3),y=mean_stats[i,:],color=["lin-pca"]));
# print("\n\n")
i=3;
j=1;
for x in eachcol(stats3)
    y=mean(x);
    # print(y);
    # print("  ")
    mean_stats[i,j]=y;
        j=j+1;
end
push!(pic,layer(x=collect(1:3),y=mean_stats[i,:],color=["tikh"]));
# print("\n\n")
i=4;
j=1;
for x in eachcol(stats4)
    y=mean(x);
    # print(y);
    # print("  ")
    mean_stats[i,j]=y;
        j=j+1;
end
push!(pic,layer(x=collect(1:3),y=mean_stats[i,:],color=["tikh-pca"]));
# print("\n\n")
i=5;
j=1;
for x in eachcol(stats5)
    y=mean(x);
    # print(y);
    # print("  ")
    mean_stats[i,j]=y;
        j=j+1;
end
push!(pic,layer(x=collect(1:3),y=mean_stats[i,:],color=["lplq"]));
# print("\n\n")
i=6;
j=1;
for x in eachcol(stats6)
    y=mean(x);
    # print(y);
    # print("  ")
    mean_stats[i,j]=y;
        j=j+1;
end
push!(pic,layer(x=collect(1:3),y=mean_stats[i,:],color=["lplq-pca"]));
# print("\n\n")
draw(SVG("mse_"*file[1:end-4]*".svg", 7inch, 7inch), pic);

# display(pic);

writedlm("mse_"*file,mean_stats ,'\t')
end
end
end
