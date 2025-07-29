using DataFrames, CSV
using LinearAlgebra
using Distributions
using Random
using ProgressMeter
using Base.Iterators
import Base: length, keys, iterate
using CairoMakie
using PairPlots

#Random.seed!(43110)

# this should be a parallel tempered sampler

lk = ReentrantLock()

length(x::Distributions.ProductNamedTupleDistribution) = length(x.dists)
keys(x::Distributions.ProductNamedTupleDistribution) = keys(x.dists)
iterate(x::Distributions.ProductNamedTupleDistribution, args...) = iterate(x.dists, args...)

function pt_mh(priors::T, log_likelihood::Function, data, nsamples :: Int, ntemps :: Int, burn=Int(2e2), Tskip::Int = 1000, tstep::Float64=1+sqrt(2/length(priors)), Tmin::Int=1) where {T <: Distribution}
    function logprior(θ)
        return logpdf(priors,θ)
    end
    samples=DataFrame([ x => Float64[] for x in keys(priors)])
    samples[:,:loglike] = Float64[]
    samples[:,:logpos] = Float64[]
    samples[:,:temp] = Float64[]
    if ntemps > Threads.nthreads()
        println("WARNING: Requested $ntemps parallel tempered chains but only $(Threads.nthreads()) threads are available. Recommend restarting Julia with more threads.")
    end
    tladder = Tmin .* tstep.^(0:(ntemps-1))
    # distribution for proposals
    σs = map(x -> x/10, map(std,priors))
    lowers = map(x->x.a,priors)
    uppers = map(x->x.b,priors)
    g(x) = product_distribution((; zip(keys(priors), [truncated(Normal(μ,σ),a,b) for (μ,σ,a,b) in zip(x,σs,lowers,uppers) ] )...))

    iter = Progress((nsamples+burn)*ntemps)
    tsamples_all = Array{DataFrame}(undef,ntemps)
    # initialize sample arrays for each chain
    for i in 1:ntemps
        tsamples=DataFrame([ x => Float64[] for x in keys(priors)])
        tsamples[:,:loglike] = Float64[]
        tsamples[:,:logpos] = Float64[]
        tsamples[:,:temp] = Float64[]
        current = rand(priors) # initialize chain
        current_ll = log_likelihood(current,data)
        current_posterior = current_ll/tladder[i] + logprior(current)
        push!(tsamples,(; current..., loglike = current_ll, logpos = current_posterior, temp=tladder[i]))
        next!(iter)
        tsamples_all[i] = tsamples
    end
    nsampled = ntemps
    while nsampled < ntemps*(nsamples + burn)
        Threads.@threads for i in 1:ntemps
            trows = nrow(tsamples_all[i])
            current = (;tsamples_all[i][end,1:end-3]...)
            current_ll = tsamples_all[i][end,:loglike]
            current_posterior = current_ll/tladder[i] + logprior(current)
            tries = 0
            while trows + Tskip >= nrow(tsamples_all[i])
                proposal = rand(g(current))
                probjump = Distributions.logpdf(g(current),proposal)
                probback = Distributions.logpdf(g(proposal),current)
                proposal_ll = log_likelihood(proposal, data)
                proposal_posterior = proposal_ll/tladder[i] + logprior(proposal)
                acceptance_prob = proposal_posterior - current_posterior + probback - probjump
                #if tries > 10000
                #    println("Extremely low acceptance rate???")
                #end
                if log(rand()) <= acceptance_prob
                    current = proposal
                    current_ll = proposal_ll
                    current_posterior = proposal_posterior
                    push!(tsamples_all[i],(; current..., loglike = current_ll, logpos = current_posterior, temp=tladder[i]))
                    next!(iter)
                    tries = 0
                else
                    tries += 1
                end
            end # while
        end # for
        # swap temps
        lock(lk) do
            nsampled += ntemps*Tskip
            for i in 1:ntemps
                n = rand(1:ntemps)
                m = rand((1:ntemps)[1:ntemps .!= n])
                r1 = (tsamples_all[m].loglike[end] - tsamples_all[n].loglike[end])/tladder[n]
                r2 = (tsamples_all[n].loglike[end] - tsamples_all[m].loglike[end])/tladder[m]
                if log(rand()) <= r1+r2
                    #println("swapping chain temps: $n:$(tladder[n]) <-> $m:$(tladder[m])")
                    temp = tladder[n]
                    tladder[n] = tladder[m]
                    tladder[m] = temp
                end
            end # for
        end # lock
    end # while
    for i in 1:ntemps
        deleteat!(tsamples_all[i],1:burn)
        lock(lk) do
            append!(samples,tsamples_all[i])
        end
    end
    finish!(iter)
    samples
end

function demo()
    # try out https://arxiv.org/abs/1008.4686
    data = CSV.read("demo_data.dat",DataFrame)
    deleteat!(data,1:4)
    priors = product_distribution((
                  m = Uniform(-1.0,5.0),
                  b = Uniform(-1000.0,1000.0)
                 ))
    function log_likelihood(θ,data)
        return sum( -0.5*(((data.y .- θ.m .* data.x .- θ.b)./data.sy).^2 .+ log(2π)) .- log.(data.sy))
    end

    results = pt_mh(priors, log_likelihood, data, 20000, 4, Int(200))
    cold_chain_results = results[results.temp .== 1.0,:]
    fig = pairplot(cold_chain_results[!,1:end-3])
    save("corner_pt.png",fig)
end
