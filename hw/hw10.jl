"""
STAT638: Homework 10 
"""

cd(@__DIR__)
using Pkg
Pkg.activate("hw10")

using Statistics 
using Distributions
using LinearAlgebra
using Plots

ys = [-1., 0., 1., 10.]

θ = Normal()
k = 300000
δ² = 1.
θs = zeros(k)

θs[1] = rand(θ)
for i in 2:k 
    J = Normal(θs[i-1], δ²)
    θʷ = rand(J)
    θᵒ = θs[i-1]
    r_log = sum(logpdf.( Normal(θʷ, 1), ys)) + logpdf(θ, θʷ) - sum(logpdf.( Normal(θᵒ, 1), ys)) - logpdf(θ, θᵒ)

    # Accept
    u_log = log(rand(Uniform(0,1)))
    if r_log > u_log 
       θs[i] = θʷ
    else 
        θs[i] = θᵒ
    end 
end



σ²ₙ = (1. + length(ys) * 1)^-1
μₙ = (length(ys)* 1/ (σ²ₙ^-1)) * mean(ys)
θ_exact = Normal(μₙ, σ²ₙ)
xs = 0:0.01:4
ys_exact = pdf.(θ_exact, xs)


function exact(θ, ys)
    p_log = -0.5 * θ^2 + (-0.5)* sum((ys .- θ).^2)
    return exp(p_log)
end

xs = 0:0.00001:4
ys_exact = [exact(x, ys) for x in xs]
ys_exact = ys_exact 

p = histogram(θs, normalize=true, xlabel="θ", ylabel="P(θ|y)", color="black", label="Metropolis")
plot!(xs, ys_exact)
