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


"""
Problem 9.1 (a)
"""

@proto struct SwimmingModel 
    S = 1000 # Number of sampling
    # Data
    y 
    n = length(y) # number of records
    # Model
    X = hcat( ones(n), collect(0:2:10) )
    p = size(X)[2]
    # Prior
    ν₀ = 1.
    σ₀² = 0.2
    β₀ = MvNormal([23., 0.], [0.1 0; 0 0.1])
end


function SSR(β, y, X)
    ssrV = (y - X*β)' * (y - X*β)
    return sum(ssrV)
end

function β_FCD(σ², m::SwimmingModel)
    Σₙ =( m.β₀.Σ^-1 + m.X' * m.X / σ²)^-1
    μₙ = Σₙ*(m.β₀.Σ^-1 * m.β₀.μ + m.X' * m.y / σ²)
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
ms = [ SwimmingModel(y = hcat(ys[i,:]), S=10000 ) for i in 1:size(ys)[1] ]
ys_pred = zeros(size(ys)[1], ms[1].S) 
X_pred = [1,12]

@time for i in eachindex(ms)
    ys_pred[i,:] = pred([1,12], ms[i]).y
end

## Plotting
p = [histogram(ys_pred[i,:], label="Swimmer $i", color="black",
    xlabel="Seconds", ylabel="Porbability", normalize= true
    ) for i in 1:size(ys)[1]]
plot(p...)


"""
Problem 9.1 (b)
"""

am = argmax(ys_pred, dims=1)

y_count = zeros(1, size(ys)[1])

for a in am 
    y_count[a[1]] += 1
end

pmax = vec(y_count ./ length(am))

## Recommendation
ds = DataFrame( Dict("Swimmer"=> collect(1:size(ys)[1]),"Pr(Y_i is max)" => pmax ))
print("$(argmax(pmax)[1]) is the most probable winner.")


"""
Problem 9.2 (a)
"""

data = readdlm("data/azdiabetes.dat")
dt = data[1:end , 1:end-1]
y = float.(dt[2:end, 2])
X = float.(dt[2:end, 1:end .!= 2])
ns = data[1,1:end-1]
ns = ns[1:end .!=2]

@proto struct DiabetesModel 
    S = 500 # Number of sampling
    # Data
    y 
    X
    n = length(y) # number of records
    p = size(X)[2]
    # Model 
    # Prior
    g = n # g prior
    ν₀ = 2.
    σ₀² = 1.
end

function β_FCD(σ², m::DiabetesModel)
    return β_FCD(σ², m.g, m.X, m.y)
end

function β_FCD(σ², g, X, y)
    Σₙ = g/(g+1) * σ² * (X'X)^-1
    μₙ = g/(g+1) * β̂(σ², y, X)
    βₙ =  MvNormal(μₙ, Hermitian(Σₙ))
    return βₙ    
end

function β̂(σ², y, X)
    return σ² * (X'X)^-1 * (X'y / σ²)
end



function σ²_FCD(m::DiabetesModel)
    return σ²_FCD(m.ν₀, m.σ₀², m.n, m.X, m.y, m.g)
end

function σ²_FCD(ν₀, σ₀², n, X, y, g)
    α = ν₀ + n / 2.
    θ = (ν₀ * σ₀² + SSR(X, y, g))/2.
    σ² = InverseGamma(α, θ)
    return σ²
end

function SSR(m::DiabetesModel)
    return SSR(m.X, m.y, m.g)
end

function SSR(X, y, g)
    return y'*(I - g/(g+1)*X*(X'X)^-1*X')*y
end


m = DiabetesModel(y=y, X=X )
σ²smp = zeros(m.S, 1)
βsmp = zeros(m.S, size(m.X)[2])

for i in 1: m.S
    σ²smp[i] = rand(σ²_FCD(m))
    βsmp[i,:] = rand(β_FCD(σ²smp[i], m))
end

ps = [histogram(βsmp[:,i], xlabel="Quantity", 
      ylabel="Probability",normalize=true, label="$(ns[i])", color="black") 
      for i in 1:m.p]
plot(ps...)


"""
Problem 9.2 (b)
"""

m = DiabetesModel(y=y, X=X,  S=1000)


function σ²_FCD(m::DiabetesModel, zs)
    Xz = @view m.X[1:end, Bool.(zs)]
    return σ²_FCD(m.ν₀, m.σ₀², m.n, Xz, m.y, m.g)
end

function β_FCD(σ², m::DiabetesModel, zs)
    Xz  = @view m.X[1:end, Bool.(zs)]
    return β_FCD(σ², m.g, Xz, m.y)
end

function y_margin(σz², m::DiabetesModel, zs)
    ν₀ = m.ν₀
    n = m.n
    y = m.y 
    g = m.g
    pz = sum(zs)
    Xz  = @view m.X[1:end, Bool.(zs)]
    ssr = SSR(Xz, y, g)

    pyl = (pz/2.)log(1. +g) + (ν₀/2.)*log(σz²) - ((ν₀+n)/2)log((ν₀*σz² + ssr))
    return pyl
end

function z_FCD(i , σz², zsmp, nSmp,m::DiabetesModel)
    zs = zsmp[nSmp,:]
    pj1 = sum(zsmp[1:nSmp, i]) / length(zsmp[1:nSmp, i])
    pj0 = 1. - pj1
    pj1_FCD_l =  pj1 * y_margin(σz², m, ones(length(zs)))
    pj0_FCD_l =  pj0 * y_margin(σz², m, zs)
    O = exp(pj0_FCD_l - pj1_FCD_l)
    return Bernoulli( 1/(1+O))
end


zsmp = ones(m.S, size(m.X)[2])
σ²smp = zeros(m.S, 1)
βsmp = zeros(m.S, size(m.X)[2])

σ²smp[1] = 0.1
# Gibbs sampling
for i in 2:m.S
    for j in Random.shuffle(1:m.p)
        zsmp[i, j] = rand(z_FCD(j, σ²smp[i-1], zsmp, i-1, m))
    end

    σ²smp[i] = rand(σ²_FCD(m, zsmp[i,:]))
    βsmp[i, Bool.(zsmp[i,:])] = rand(β_FCD(σ²smp[i], m, zsmp[i,:]))
end




sum(zsmp, dims=1)/size(zsmp)[1]
prB = 1. .- vec(sum(zsmp, dims=1))./size(zsmp)[1]
DataFrame(Dict( "Bi"=> ns, "Pr(Bi != 0 |y)"=> prB ))


inds = prB .>= 0.5
b_select = prB[inds]

ps = [histogram(βsmp[:, i], xlabel="Quantity", 
      ylabel="Probability",normalize=true, label="$(ns[i])", color="black") 
      for i in  findall(inds .== 1)]
plot(ps...)




DataFrame(Dict("Bi"=> ns[inds], 
               "Confidence"=> [quantile(βsmp[:,i], [0.25, 0.975]) for i in  findall(inds .== 1)]))
