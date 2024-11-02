using DataFrames, CSV
using LinearAlgebra
using Distributions
using Random
using ProgressMeter
using Base.Iterators
import Base.rand
using CairoMakie
using PairPlots

#Random.seed!(43110)

# this should be a parallel tempered sampler

lk = ReentrantLock()

rand(x::NamedTuple) = map(rand,x)
# assume distributions are boring products
Distributions.pdf(d::NamedTuple,p::NamedTuple) = prod([Distributions.pdf(x,y) for (x,y) in zip(d,p)])
logpdf(d::NamedTuple,p::NamedTuple) = sum([Distributions.logpdf(x,y) for (x,y) in zip(d,p)])

function pt_mh(priors :: NamedTuple, log_likelihood::Function, data, nsamples :: Int,num_chains :: Int,burn=Int(2e2),Tskip::Int = 1000,tstep::Float64=1+sqrt(2/length(priors)),Tmin::Int=1)
    function logprior(θ)
        return logpdf(priors,θ)
    end
    samples=DataFrame([ x => Float64[] for x in keys(priors)])
    samples[:,:loglike] = Float64[]
    samples[:,:logpos] = Float64[]
    samples[:,:temp] = Float64[]
    if num_chains > Threads.nthreads()
        println("WARNING: Requested $num_chains parallel tempered chains but only $(Threads.nthreads()) threads are available. Recommend restarting Julia with more threads.")
    end
    tladder = Tmin .* tstep.^(0:(num_chains-1))
    # distribution for proposals
    σs = map(x -> x/10, map(std,priors))
    lowers = map(x->x.a,priors)
    uppers = map(x->x.b,priors)
    g(x) = (; zip(keys(priors), [truncated(Normal(μ,σ),a,b) for (μ,σ,a,b) in zip(x,σs,lowers,uppers) ] )...)

    iter = Progress((nsamples+burn)*num_chains)
    tsamples_all = Array{DataFrame}(undef,num_chains)
    # initialize sample arrays for each chain
    for i in 1:num_chains
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
    nsampled = num_chains
    while nsampled < num_chains*(nsamples + burn)
        Threads.@threads for i in 1:num_chains
            trows = nrow(tsamples_all[i])
            current = (;tsamples_all[i][end,1:end-3]...)
            current_ll = tsamples_all[i][end,:loglike]
            current_posterior = current_ll/tladder[i] + logprior(current)
            tries = 0
            while trows + Tskip >= nrow(tsamples_all[i])
                proposal = rand(g(current))
                probjump = Distributions.pdf(g(current),proposal)
                probback = Distributions.pdf(g(proposal),current)
                proposal_ll = log_likelihood(proposal, data)
                proposal_posterior = proposal_ll/tladder[i] + logprior(proposal)
                acceptance_prob = proposal_posterior - current_posterior + log(probback/probjump)
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
            nsampled += num_chains*Tskip
            for i in 1:num_chains
                n = rand(1:num_chains)
                m = rand((1:num_chains)[1:num_chains .!= n])
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
    for i in 1:num_chains
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
    priors = (
                  m = Uniform(-1.0,5.0),
                  b = Uniform(-1000.0,1000.0)
                 )
    function log_likelihood(θ,data)
        return sum( -0.5*(((data.y .- θ.m .* data.x .- θ.b)./data.sy).^2 .+ log(2π)) .- log.(data.sy))
    end

    results = pt_mh(priors,log_likelihood,data,20000,4,Int(200))
    valid_results = results[results.temp .== 1.0,:]
    fig = pairplot(valid_results[!,1:end-3])
    save("corner_pt.png",fig)
end
