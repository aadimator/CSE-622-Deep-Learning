### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 373d0f40-265b-11eb-24a5-55e733605914
begin
	using Pkg
    Pkg.activate(pwd())
end

# ╔═╡ 864115a0-20d4-11eb-16ff-71d198f9342e
begin
	using Plots
	using LinearAlgebra
	using Distributions
	
	gr();
end

# ╔═╡ a33c14a0-20c2-11eb-0bf8-837173d9d91c
md"""
# Assignment 2: Implement Gradient Descent
### Aadam (CS1945)
### CSE 662 - Deep Learning by Dr. Masroor Hussain

---

## Assignment Statement:
Write down the code to optimize $f(x) = \frac{1}{2} \|Ax-b\|^2_2$ using gradient descent method in C/C++/Python/Matlab. See equation 4.21.

---
"""

# ╔═╡ cf5c9eb0-20c2-11eb-05ba-2f5dd23fdaf4
md"""
# Gradient Descent
Gradient Descent is a well-known iterative optimization algorithm, widely used in Machine Learning and Deep Learning. It is a way to minimize an objective function $f(x)$ parameterized by a model's parameters $x$ by updating the parameters in the opposite direction of the gradient of the objective function $\nabla_xf(x)$ w.r.t the parameters.

For this assignment, we'll use the following algorithm, from the Deep Learning book:
![algo](https://i.imgur.com/XSwselr.png)
"""

# ╔═╡ b6ed7820-20ce-11eb-0584-9130387e39ef
md"""
# Solution:
"""

# ╔═╡ 2782f610-262d-11eb-0017-6314af9f27a0
md"""
First, we'll randomly initialize $A$ (a matrix) and $b$ (a vector) from a normal distribution. $A$ will be of size $m \times n$ and $b$ is a vector of length $m$.
"""

# ╔═╡ 667a0680-2595-11eb-2859-258f6407b790
rnorm = Normal();

# ╔═╡ 39217590-262d-11eb-0858-5106bc2498d8
m, n = 2, 5;

# ╔═╡ e009fab0-2594-11eb-1711-475e4d2bd945
𝐀 = rand(rnorm, m, n)

# ╔═╡ 7e8c2280-2595-11eb-1eed-cd25325faa84
𝒃 = rand(rnorm, m)

# ╔═╡ b9f447b0-262d-11eb-0444-4582e4bb3e37
md"""
Now, let's define our objective function, i.e. $f(x) = \frac{1}{2} \|Ax-b\|^2_2$.
"""

# ╔═╡ 0a73bcbe-2584-11eb-398f-57c8bc991795
𝑓(𝒙) = 1/2 * norm(𝐀 * 𝒙 .- 𝒃)^2

# ╔═╡ 0a59cc20-2584-11eb-3ac8-d3f4c1a325ec
𝑓(0)

# ╔═╡ e3246890-262d-11eb-1ab3-eb263ec2ae51
md"""
The gradient of this function would be:
$
\nabla_xf(x) = A^T(Ax - b) = A^TAx - A^Tb
$
"""

# ╔═╡ 0a3db8a0-2584-11eb-0508-d935b4b32358
∇ₓ𝑓(𝒙) = 𝐀'𝐀 * 𝒙 .- 𝐀'𝒃

# ╔═╡ 3f354f30-25c7-11eb-2d7b-29d7a44aadd7
∇ₓ𝑓(0)

# ╔═╡ 81edab80-262e-11eb-129c-55f41ac9c662
md"""
Let's define our step size $(\epsilon)$ and tolerance $(\delta)$.
"""

# ╔═╡ 09bdda40-2584-11eb-0a1d-5535b8cbfffb
ϵ = 0.01;

# ╔═╡ 10c38d80-2598-11eb-1c62-3bc6c09ef195
δ = 1e-10;

# ╔═╡ 3e2aa142-265c-11eb-2731-5bb42f1843f5
md"""
Here, $x$ is a vector of length $n$, starting from an arbitrary value, randomly taken from a normal distribution.
"""

# ╔═╡ 5d45dce0-25b5-11eb-3ea7-4964249ce9a5
𝒙 = rand(rnorm, n)

# ╔═╡ 10df79f0-2598-11eb-39a5-476ac6e96ff5
function gradient_descent(𝒙)
	i = 0
	fxvals = []
	while norm(∇ₓ𝑓(𝒙)) > δ && i < 10000
		push!(fxvals, 𝑓(𝒙))
		𝒙 = 𝒙 - ϵ * (∇ₓ𝑓(𝒙))
		i += 1
		println("$i : $𝒙")
	end
	𝒙, fxvals
end

# ╔═╡ 58c08130-25c8-11eb-0efd-81830bcfaecd
minx, fxvals = gradient_descent(𝒙);

# ╔═╡ 5a03b4c0-25b6-11eb-3ad0-753d1aec5cfb
𝑓(minx)

# ╔═╡ cf29ec50-265c-11eb-1796-53065b4536e4
plot(
	fxvals, 
	title = "Value of f(x)", 
	label = "f(x)", 
	xaxis="Steps", 
	yaxis="f(x)", 
	lw = 2
)

# ╔═╡ 645934a0-20fb-11eb-1750-cf7ca28ab9e6
md"""
## Conclusion
In this notebook, we showed how we can use **Gradient Descent** algorithm to find the optimum value of $x$ that minimizes the function $f(x) = \frac{1}{2} \|Ax-b\|^2_2$.
"""

# ╔═╡ Cell order:
# ╟─a33c14a0-20c2-11eb-0bf8-837173d9d91c
# ╟─cf5c9eb0-20c2-11eb-05ba-2f5dd23fdaf4
# ╟─b6ed7820-20ce-11eb-0584-9130387e39ef
# ╠═373d0f40-265b-11eb-24a5-55e733605914
# ╠═864115a0-20d4-11eb-16ff-71d198f9342e
# ╟─2782f610-262d-11eb-0017-6314af9f27a0
# ╠═667a0680-2595-11eb-2859-258f6407b790
# ╠═39217590-262d-11eb-0858-5106bc2498d8
# ╠═e009fab0-2594-11eb-1711-475e4d2bd945
# ╠═7e8c2280-2595-11eb-1eed-cd25325faa84
# ╟─b9f447b0-262d-11eb-0444-4582e4bb3e37
# ╠═0a73bcbe-2584-11eb-398f-57c8bc991795
# ╠═0a59cc20-2584-11eb-3ac8-d3f4c1a325ec
# ╟─e3246890-262d-11eb-1ab3-eb263ec2ae51
# ╠═0a3db8a0-2584-11eb-0508-d935b4b32358
# ╠═3f354f30-25c7-11eb-2d7b-29d7a44aadd7
# ╟─81edab80-262e-11eb-129c-55f41ac9c662
# ╠═09bdda40-2584-11eb-0a1d-5535b8cbfffb
# ╠═10c38d80-2598-11eb-1c62-3bc6c09ef195
# ╟─3e2aa142-265c-11eb-2731-5bb42f1843f5
# ╠═5d45dce0-25b5-11eb-3ea7-4964249ce9a5
# ╠═10df79f0-2598-11eb-39a5-476ac6e96ff5
# ╠═58c08130-25c8-11eb-0efd-81830bcfaecd
# ╠═5a03b4c0-25b6-11eb-3ad0-753d1aec5cfb
# ╠═cf29ec50-265c-11eb-1796-53065b4536e4
# ╟─645934a0-20fb-11eb-1750-cf7ca28ab9e6
