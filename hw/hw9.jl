using Pkg
Pkg.activate("hw9")
using Distributions
using DataFrames
using Plots
using DelimitedFiles
using LinearAlgebra
using Statistics
using ProtoStructs
import Random
Random.seed!(2022)


@proto struct SwimmingModel 
    S = 1000 # Number of sampling
    # Data
    y 
    n = length(y) # number of records
    # Model
    X = hcat( ones(n), collect(0:2:10) )
    p = size(X)[2]
    # Prior
    β₀ = MvNormal([23., 0.], [0.1 0; 0 0.1])
    ν₀ = 1.
    σ₀² = 0.2
end


function SSR(β, y, X)
    ssrV = (y - X*β)' * (y - X*β)
    return sum(ssrV)
end

function β_FCD(σ², m::SwimmingModel)
    Σₙ =( m.β₀.Σ^-1 + m.X' * m.X / σ²)^-1
    μₙ = Σₙ*(m.β₀.Σ^-1 * m.β₀.μ + m.X' * m.y / σ²)
    @show Σₙ
    return MvNormal(vec(μₙ), Hermitian(Σₙ))
end

function σ²_FCD(β, m::SwimmingModel)
    α = (m.ν₀ + m.n)/2
    θ = (m.ν₀*m.σ₀²) + SSR(β, m.y, m.X)
    return InverseGamma(α, θ)
end

function pred(X, m::SwimmingModel)
    # Sampling vector
    βsmp = zeros(m.S, length(m.β₀.μ))
    σ²smp = zeros(m.S)
    y = zeros(m.S)
    # Init
    βsmp[1,:] = rand(m.β₀)
    σ²smp[1] = m.σ₀²
    y[1] = m.y[1]
    for i in 2:m.S 
        βsmp[i,:] = rand(β_FCD(σ²smp[i-1], m))
        σ²smp[i] = rand(σ²_FCD(βsmp[i-1,:], m))

        # Predict 
        y[i] = βsmp[i,:]' * X + rand(Normal(0., σ²smp[i]))
    end

    return (y=y, β=βsmp, σ²=σ²smp)
end

ys = readdlm("data/swim.dat")
j_swim = 1
m = SwimmingModel(y = hcat(ys[j_swim,:]))


# Sampling vector
βsmp = zeros(m.S, length(m.β₀.μ))
σ²smp = zeros(m.S)
y = zeros(m.S)
# Init
βsmp[1,:] = rand(m.β₀)
σ²smp[1] = m.σ₀²
y[1] = m.y[1]
for i in 2:m.S 
    βsmp[i,:] = rand(β_FCD(σ²smp[i-1], m))
    σ²smp[i] = rand(σ²_FCD(βsmp[i-1,:], m))

    # Predict 
    @show βsmp[i,:] * [1 12] 
    y[i] = βsmp[i,:]' * [1, 12] + rand(Normal(0., σ²smp[i]))
end

pred([1,12], m)

## Plotting
histogram(y)
