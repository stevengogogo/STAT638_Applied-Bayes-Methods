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

th = 0.
for i in 1:k 
    θᵒ = th
    J = Normal(θᵒ, δ²)
    θʷ = rand(J)
    r_log = sum(logpdf.( Normal(θʷ, 1), ys) .- logpdf.( Normal(θᵒ, 1), ys)) + logpdf(θ, θʷ) - logpdf(θ, θᵒ)

    # Accept
    u_log = log(rand(Uniform()))
    if r_log > u_log 
        θs[i] = θʷ
    else 
        θs[i] = θᵒ
    end 
    th = θs[i]
end


# Exact PDF
n = length(ys)
τₙ =( 1 / (n/ 1 + 1/ 1)^0.5)
μₙ = mean(ys) * (n / 1) / (n/ 1 + 1/1) + 0 
θ_exact = Normal(μₙ, τₙ )
xs = 0:0.01:4
ys_exact = pdf.(θ_exact, xs)

# Display
p = histogram(θs, normalize=:pdf, xlabel="θ", ylabel="P(θ|y)", label="Metropolis", title="Normal Model")
plot!(xs, ys_exact)

"""
"""