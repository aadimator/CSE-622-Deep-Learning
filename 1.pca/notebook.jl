### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ dcfde69e-20c7-11eb-0108-dbd2f5cb0098
begin
	using Pkg
    Pkg.activate(pwd())
	Pkg.instantiate()
end

# ╔═╡ 864115a0-20d4-11eb-16ff-71d198f9342e
begin
	using CSV
	using DataFrames
	using RDatasets
	using StatsBase
	using Plots
	using LinearAlgebra
	import Statistics: covm
end

# ╔═╡ a33c14a0-20c2-11eb-0bf8-837173d9d91c
md"""
# Assignment 1: Implement PCA
### Aadam (CS1945)
### CSE 662 - Deep Learning by Dr. Masroor Hussain

---

## Assignment Statement:
Write the code in C, Python or Matlab for $l=1$ PCA as discussed in class of Chapter no. 2. Five bonus points for $1 =< l < n$

---
"""

# ╔═╡ cf5c9eb0-20c2-11eb-05ba-2f5dd23fdaf4
md"""
# Principal Component Analysis (PCA)
PCA is a technique that is used to derive an orthogonal projection to convert a given set of observations to linearly uncorrelated variables, called **principal components**.

One of the most widely used application of PCA is **dimensionality reduction**, which helps us out, when training our models on large datasets, by minimizing noise and redundancy in the dataset. Basic idea of the dimensionality reduction is to represent an $M$-dimensional data into an $N$-dimensional subspace, where $N < M$. This would result in data loss, but our aim is to minimize it as much as possible. For that purposes, we try to find some **principal components** in the data that can represent the features as a linear combination, without loosing much information.

PCA uses **covariance matrix** to analyze:
- variance of each feature, showing if a feature is relevant or pure noise.
- linear relationship strength between pairs of features, spotting redundant features.

There are two main approaches to identify principal components:
1. Calculate the **eigenvectors** of the covariance matrix.
2. Calculate the **single value decomposition** of the covariance matrix.

Although **SVD** has higher numerical accuracy, it has lower running time, as compared to the **eigenvector decomposition** method.

For this assignment, we'll use **Eigendecomposition** method to find principal components. 
"""

# ╔═╡ b6ed7820-20ce-11eb-0584-9130387e39ef
md"""
# Solution:
First, let's create a dummy dataset.
"""

# ╔═╡ 85bf8990-20d4-11eb-1a7b-e35d9e62f35a
gr();

# ╔═╡ f5d95832-20ea-11eb-2828-39ebb61c877f
md"""
### Utility functions
"""

# ╔═╡ 0024d7b0-20eb-11eb-2959-af3584eb7750
begin
	centralize(x::AbstractVector, m::AbstractVector) = (isempty(m) ? x : x - m)
	centralize(x::AbstractMatrix, m::AbstractVector) = (isempty(m) ? x : x .- m)

	decentralize(x::AbstractVector, m::AbstractVector) = (isempty(m) ? x : x + m)
	decentralize(x::AbstractMatrix, m::AbstractVector) = (isempty(m) ? x : x .+ m)
	
	transform(P, x, mv) = transpose(P) * centralize(x, mv)
	reconstruct(P, y, mv) = decentralize(P * y, mv)
	
	# extract k values/vectors
	function extract_kv(fac, ord, k)
		si = ord[1:k]
		vals = fac.values[si]
		vecs = fac.vectors[:, si]
		return (vals, vecs)
	end
end

# ╔═╡ 85976820-20d4-11eb-2afa-47a833fdccc4
md"""
## Input Data
For this example, we'll use **iris** dataset.
"""

# ╔═╡ dcce4b20-20c7-11eb-3cb1-852a46b75c31
iris = dataset("datasets", "iris") # DataFrame(rand(100, 10))

# ╔═╡ dcb4a8a2-20c7-11eb-3a1a-9fb620e84e57
size(iris)

# ╔═╡ dc975ca0-20c7-11eb-0a2b-b9e470c6e311
describe(iris)

# ╔═╡ 9241d120-20d2-11eb-0b30-25249995c00c
md"""
#### Let's create separate train and test datasets.
"""

# ╔═╡ b2a8ccc0-20d2-11eb-28eb-87dc018ae352
Xtrain = Array(iris[1:2:end,1:4])';

# ╔═╡ afb22c8e-20d3-11eb-339a-09f39705a02f
Ytrain = Array(iris[1:2:end,5]);

# ╔═╡ afe6aa10-20d3-11eb-03a4-89a46ade9feb
Xtest = Array(iris[2:2:end,1:4])';

# ╔═╡ b0004c90-20d3-11eb-2b38-452b177600d8
Ytest = Array(iris[2:2:end,5]);

# ╔═╡ b02fe810-20d3-11eb-21f4-f9939c117c12
md"""
Suppose `Xtrain` and `Xtest` are training and testing data matrix, with each observation in a column.

## Data Normalization
Now, we need to normalize the data.

"""

# ╔═╡ 399f2a20-20d4-11eb-1c3a-e76ac9707e29
mean_iris = vec(mean(Xtrain, dims=2))

# ╔═╡ c58d8a10-20e1-11eb-301a-eb715e5926d8
md"""
## Covariance Matrix Computation
Covariance matrix is an $N \times N$ symmetric matrix, where $N$ is the number of dimensions. This tells us about the relation between any two features, and the variation of data from the mean.
"""

# ╔═╡ 39b65ba2-20d4-11eb-11bd-7daa396810ac
C_iris = covm(Xtrain, mean_iris, 2);

# ╔═╡ 6493e090-20e3-11eb-0dec-0bdef8eb1c6a
DataFrame(C_iris)

# ╔═╡ c308e590-20e2-11eb-0ddb-055f8bdedb85
md"""
## Eigendecomposition of Covariance Matrix
We need to find the eigenvalues and eigenvectors of the covariance matrix in order to find the principal components of the data. These components are created in such a way that they are uncorrelated, and the first component contains the maximum amount of information possible, and the second contains the maximum remaining, and so on.
"""

# ╔═╡ 39e24da0-20d4-11eb-10e5-8320e876fb74
eg = eigen(Symmetric(C_iris));

# ╔═╡ b04fa510-20d3-11eb-0d40-21f2bf50e115
eg_values = eg.values

# ╔═╡ f010e210-20e5-11eb-1952-37e6ed84ea22
eg_vectors = eg.vectors

# ╔═╡ 35710110-20e5-11eb-0ad2-ef911c8d78bc
md"""
Let's sort the eigenvalues in reverse order, so the component which contains the most information is at the front.
"""

# ╔═╡ b1d8dc20-20de-11eb-1364-59fc5f488736
ord = sortperm(eg_values; rev=true)

# ╔═╡ 6f9d6dfe-20e5-11eb-2745-e95f930eaf88
md"""
Let's see how much information each component contains. For that, we need to find the total variance of the input, and then divide the principal component's variance with it. We'll plot the values for better visualization.
"""

# ╔═╡ b211ed80-20de-11eb-3abc-7f0b887915a5
tvar = sum(eg_values)

# ╔═╡ 9fefec80-20e6-11eb-3e16-e5f17e1d9560
eg_values[ord] ./ tvar * 100

# ╔═╡ b25a1a10-20de-11eb-3bcb-fbe70a5b974d
bar(eg_values[ord] ./ tvar * 100, title="Scree Plot")

# ╔═╡ b26da210-20de-11eb-026b-8fc4efce305c
md"""
By looking at the above graph, we can see that the first component, $PC 1$, contains more than $90 \%$ of the information, while $PC 2$ contains less than $5 \%$.

Even though many believe $PCA$ to be a dimensionality reduction algorithm, primarily it's a data transformation algorithm. It just makes our data amenable to data reduction, as shown above. Here, as most of the information is contained in the first couple of principal components, we can decide to keep only those componenets and discard the rest, resulting in a reduction in the dimensionality of the dataset.

### $l = 1$:
This means that we only take the first PC and discard the rest.
"""

# ╔═╡ d6429920-20cf-11eb-0bf1-8f74c17dc643
v, P = extract_kv(eg, ord, 1)

# ╔═╡ 72f60020-20eb-11eb-049a-d14c26e12ddb
md"""
This returns the eigen values, as well as the project matrix $P$. We can use $P$ to transform dataset. Transforming the `Xtest` using $P$ would result in:
"""

# ╔═╡ d628f6a0-20cf-11eb-11cc-01f1ca2b6f13
tr_data = transform(P, Xtest, mean_iris)

# ╔═╡ 6ae889c0-20eb-11eb-3ecb-4f13647511c3
md"""
which is a $1 \times 75$ array, instead of the original $4 \times 75$.  

Let's try to reconstruct the original data from the transformed data using $P$.
"""

# ╔═╡ 1081c390-20f8-11eb-3e81-133df7444871
recon_data = reconstruct(P, tr_data, mean_iris);

# ╔═╡ 3313dc90-20f8-11eb-27eb-e9ea7e6a3805
md"""
Printing the first 5 observations from both the reconstructed data as well as the original data.
"""

# ╔═╡ d5e95590-20cf-11eb-0d5f-e3e5f2be14c0
DataFrame(recon_data)

# ╔═╡ d5b26710-20cf-11eb-0f4c-7f81055af3e3
DataFrame(Xtest[:, 1:5])

# ╔═╡ 4cd4fb30-20c3-11eb-0631-a9c8840a4cf9
md"""
As can be seen, the reconstructed data is fairly close to the original data, although not exactly the same because this is a lossy method. Even though we retained around $92\%$ information, we still lost some information by reducing the dimensions.

We can retain more information by increasing $l$, i.e. increasing the number of output dimensions.

## $1 <= l < n$
Instead of taking only $PC 1$, we will take the first three PCs.
"""

# ╔═╡ 740d6e20-20ec-11eb-1b12-1d614d3efb08
v3, P3 = extract_kv(eg, ord, 3)

# ╔═╡ 7466b1b2-20ec-11eb-1a71-e7061e6a7ad6
md"""
Again, let's first transform and reconstruct the data using `P3`.
"""

# ╔═╡ 747f1bb0-20ec-11eb-3cbf-a3a8d7a64563
tr3_data = transform(P3, Xtest, mean_iris);

# ╔═╡ 65ce7a10-20fc-11eb-1d02-f57de17db24e
size(tr3_data)

# ╔═╡ 7493b520-20ec-11eb-0e19-57cbd43d09af
recon3_data = reconstruct(P3, tr3_data, mean_iris);

# ╔═╡ 74a9ae20-20ec-11eb-1e5d-7984c0a399f3
DataFrame(recon3_data)

# ╔═╡ ff1b4e90-20f8-11eb-3924-2fa0c61677fd
DataFrame(Xtest[:, 1:5])

# ╔═╡ 053f9290-20f9-11eb-1cb6-6f5c7b26e279
md"""
We can see that using $l = 3$ gives a better approximation of the original data as compared to $l = 1$.

## Visualization
Let's visualize the first 3 principal components in a 3D plot.
"""

# ╔═╡ 9599eb10-20f9-11eb-044b-cd71da0c30a3
begin
	# group by labels, for color coding
	setosa = tr3_data[:,Ytest.=="setosa"];
	versicolor = tr3_data[:,Ytest.=="versicolor"];
	virginica = tr3_data[:,Ytest.=="virginica"];
	
	p = scatter(setosa[1,:],setosa[2,:],setosa[3,:],marker=:circle,linewidth=0)
	scatter!(versicolor[1,:],versicolor[2,:],versicolor[3,:],marker=:circle,linewidth=0)
	scatter!(virginica[1,:],virginica[2,:],virginica[3,:],marker=:circle,linewidth=0)
	plot!(p,xlabel="PC1",ylabel="PC2",zlabel="PC3")
end

# ╔═╡ 645934a0-20fb-11eb-1750-cf7ca28ab9e6
md"""
## Conclusion
In this notebook, we showed how we can use `PCA` to find principal components of some data, and then use those for dimensionality reduction without severly affecting and losing the information.
"""

# ╔═╡ 4d526890-20c3-11eb-082b-e912d7c9b3e2
md"""
# Resources

- [StatQuest: Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ&t=458s&ab_channel=StatQuestwithJoshStarmer)
- [Data Analysis 6: Principal Component Analysis (PCA) - Computerphile](https://www.youtube.com/watch?v=TJdH6rPA-TI&ab_channel=Computerphile)
- [Making sense of principal component analysis, eigenvectors & eigenvalues](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579)
- [A STEP BY STEP EXPLANATION OF PRINCIPAL COMPONENT ANALYSIS](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
- [Machine Learning — Singular Value Decomposition (SVD) & Principal Component Analysis (PCA)](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)
- [Relationship between SVD and PCA. How to use SVD to perform PCA?](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
"""

# ╔═╡ Cell order:
# ╟─a33c14a0-20c2-11eb-0bf8-837173d9d91c
# ╟─cf5c9eb0-20c2-11eb-05ba-2f5dd23fdaf4
# ╟─dcfde69e-20c7-11eb-0108-dbd2f5cb0098
# ╟─b6ed7820-20ce-11eb-0584-9130387e39ef
# ╠═864115a0-20d4-11eb-16ff-71d198f9342e
# ╟─85bf8990-20d4-11eb-1a7b-e35d9e62f35a
# ╟─f5d95832-20ea-11eb-2828-39ebb61c877f
# ╠═0024d7b0-20eb-11eb-2959-af3584eb7750
# ╟─85976820-20d4-11eb-2afa-47a833fdccc4
# ╠═dcce4b20-20c7-11eb-3cb1-852a46b75c31
# ╠═dcb4a8a2-20c7-11eb-3a1a-9fb620e84e57
# ╠═dc975ca0-20c7-11eb-0a2b-b9e470c6e311
# ╟─9241d120-20d2-11eb-0b30-25249995c00c
# ╠═b2a8ccc0-20d2-11eb-28eb-87dc018ae352
# ╠═afb22c8e-20d3-11eb-339a-09f39705a02f
# ╠═afe6aa10-20d3-11eb-03a4-89a46ade9feb
# ╠═b0004c90-20d3-11eb-2b38-452b177600d8
# ╟─b02fe810-20d3-11eb-21f4-f9939c117c12
# ╠═399f2a20-20d4-11eb-1c3a-e76ac9707e29
# ╟─c58d8a10-20e1-11eb-301a-eb715e5926d8
# ╠═39b65ba2-20d4-11eb-11bd-7daa396810ac
# ╟─6493e090-20e3-11eb-0dec-0bdef8eb1c6a
# ╟─c308e590-20e2-11eb-0ddb-055f8bdedb85
# ╠═39e24da0-20d4-11eb-10e5-8320e876fb74
# ╠═b04fa510-20d3-11eb-0d40-21f2bf50e115
# ╠═f010e210-20e5-11eb-1952-37e6ed84ea22
# ╟─35710110-20e5-11eb-0ad2-ef911c8d78bc
# ╠═b1d8dc20-20de-11eb-1364-59fc5f488736
# ╟─6f9d6dfe-20e5-11eb-2745-e95f930eaf88
# ╠═b211ed80-20de-11eb-3abc-7f0b887915a5
# ╟─9fefec80-20e6-11eb-3e16-e5f17e1d9560
# ╠═b25a1a10-20de-11eb-3bcb-fbe70a5b974d
# ╟─b26da210-20de-11eb-026b-8fc4efce305c
# ╠═d6429920-20cf-11eb-0bf1-8f74c17dc643
# ╟─72f60020-20eb-11eb-049a-d14c26e12ddb
# ╠═d628f6a0-20cf-11eb-11cc-01f1ca2b6f13
# ╟─6ae889c0-20eb-11eb-3ecb-4f13647511c3
# ╠═1081c390-20f8-11eb-3e81-133df7444871
# ╟─3313dc90-20f8-11eb-27eb-e9ea7e6a3805
# ╠═d5e95590-20cf-11eb-0d5f-e3e5f2be14c0
# ╠═d5b26710-20cf-11eb-0f4c-7f81055af3e3
# ╟─4cd4fb30-20c3-11eb-0631-a9c8840a4cf9
# ╠═740d6e20-20ec-11eb-1b12-1d614d3efb08
# ╟─7466b1b2-20ec-11eb-1a71-e7061e6a7ad6
# ╠═747f1bb0-20ec-11eb-3cbf-a3a8d7a64563
# ╠═65ce7a10-20fc-11eb-1d02-f57de17db24e
# ╠═7493b520-20ec-11eb-0e19-57cbd43d09af
# ╠═74a9ae20-20ec-11eb-1e5d-7984c0a399f3
# ╠═ff1b4e90-20f8-11eb-3924-2fa0c61677fd
# ╟─053f9290-20f9-11eb-1cb6-6f5c7b26e279
# ╟─9599eb10-20f9-11eb-044b-cd71da0c30a3
# ╟─645934a0-20fb-11eb-1750-cf7ca28ab9e6
# ╟─4d526890-20c3-11eb-082b-e912d7c9b3e2
