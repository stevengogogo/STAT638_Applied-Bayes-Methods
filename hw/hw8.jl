using Pkg
Pkg.activate("hw8")
using Distributions
using DataFrames
using Turing
using Plots
using DelimitedFiles
using LinearAlgebra
using Statistics
using Turing

dsch = Dict()
nsch = 8
for i in 1:nsch
   dsch[i] = readdlm("data/school$i.dat")
end

# Prior
μ0 = 7.
γ0² = 5.
τ0² = 10.
η0 = 2.
σ0² = 15.
ν0 = 2.

# Data
ns = [ length(dsch[i]) for i in 1:nsch]
n = sum(ns)
m = length(dsch)
ȳs = [mean(dsch[i]) for i in 1:nsch]
s²s = [ (ns[i] - 1)^-1 * sum( (dsch[i] .- ȳs[i]).^2) for i in 1:nsch]
# posterior

function τ²_pos(m, η0, θv, μ, τ0²)
    ths² = sum([ (θ- μ)^2 for θ in θv])
    α = (m + η0)* 0.5
    β = (ths² + η0*τ0²)/2
    return InverseGamma(α, β)
end

function σ²_pos(n, ν0, σ0², ns, s²s, ȳs, θs)
    α = (n+ν0)/2
    β =( sum((ns .- 1) .* s²s + ns .* (ȳs .- θs).^2) + ν0*σ0²)/2
    return InverseGamma(α, β)
end

function μ_pos(m, τ², θs, γ0², μ0)
    γm² = (m/τ² + 1/γ0²)^-1
    θ̄ = mean(θs)

    a = γm²*(m*θ̄/τ² + μ0/τ²)
    return Normal(a, γm²)
end

function θ_pos(τ², ȳ, n, σ², μ)
    τ̃² = (n/σ² + 1/τ²)^-1
    a = τ̃²*(n*ȳ/σ² + μ/τ²)
    return Normal(a, τ̃²)
end

# Sampling
smp = 1000
τ²s = zeros(smp)
σ²s = zeros(smp)
μs = zeros(smp)
θs = zeros(smp, m)


σ²s[1] = rand(InverseGamma(ν0/2, ν0*σ0²/2))
τ²s[1] = rand(InverseGamma(η0 /2, η0 *τ0²/2))
μs[1] = rand(Normal(μ0, γ0²))
θs[1,:] = [rand(θ_pos(τ²s[1], ȳs[i], ns[i], σ²s[1], μs[1])) for i in 1:m]

for s in 2:smp
    σ²s[s] = rand(σ²_pos(n, ν0, σ0², ns, s²s, ȳs, θs[s,:]))
    τ²s[s] = rand(τ²_pos(m, η0, θs[s,:], μs[s], τ0²))
    θs[s,:] = [rand(θ_pos(τ²s[s], ȳs[i], ns[i], σ²s[s], μs[s])) for i in 1:m]
    μs[s] = rand(μ_pos(m, τ²s[s], θs[s,:], γ0², μ0))
end

plot(σ²s)