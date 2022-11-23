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
using KernelDensity

ys = [-1., 0., 1., 10.]



θ = Normal()
k = 300000
δ² = 1.

function sampling_ma(like_dist, θ, k, δ²)
    θs = zeros(k)
    th = 0.
    for i in 1:k 
        θᵒ = th
        J = Normal(θᵒ, δ²)
        θʷ = rand(J)
        r_log = sum(logpdf.( like_dist(θʷ, 1), ys) .- logpdf.( like_dist(θᵒ, 1), ys)) + logpdf(θ, θʷ) - logpdf(θ, θᵒ)

        # Accept
        u_log = log(rand(Uniform()))
        if r_log > u_log 
            θs[i] = θʷ
        else 
            θs[i] = θᵒ
        end 
        th = θs[i]
    end
    return θs
end

θs_n = sampling_ma(Normal, θ, k, δ²)

# Exact PDF
n = length(ys)
τₙ =( 1 / (n/ 1 + 1/ 1)^0.5)
μₙ = mean(ys) * (n / 1) / (n/ 1 + 1/1) + 0 
θ_exact = Normal(μₙ, τₙ )
xs = 0:0.01:4
ys_exact = pdf.(θ_exact, xs)

# Display
p = histogram(θs, normalize=:pdf, xlabel="θ", ylabel="P(θ|y)", label="Metropolis", title="Normal Model")
plot!(p, xs, ys_exact)

"""
(2)
"""

θs_c = sampling_ma(Cauchy, θ, k, δ²)
p2 = histogram(θs, normalize=:pdf, xlabel="θ", ylabel="P(θ|y)", label="Metropolis", title="Cauchy Model")

display(p2)


"""
(3)
"""
Un = kde(θs_n)
Uc = kde(θs_c)
pdf_n = [pdf(Un, x) for x in xs]
pdf_c = [pdf(Uc, x) for x in xs]
p3 = plot()
plot!(p3, xs, pdf_n, label="Normal")
plot!(p3, xs, pdf_c, label="Cauchy")
plot!(p3, xs, ys_exact, label="Exact PDF (Normal)")
display(p3)