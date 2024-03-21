# some-lisa-tools-julia
A very small collection of LISA-relevant Julia codes. Includes contributions from Maria Jose Bustamante Rosell (@majoburo) and Robbie Rosati (@rjrosati)

# Setting up
First [install Julia](https://julialang.org/downloads/), then clone this repo.
Open a terminal in the cloned repo's folder. Start Julia with several threads (for the demo to work, we'll want 4).

```shell
$ julia -t 4
```
As a small aside, the Julia workflow is a little different than python -- because Julia uses just-in-time compilation, you'll avoid excessive recompilation if you leave a julia REPL running between editing and rerunning code.

## Activating the environment
Next we'll want to activate the environment in this folder
```julia-repl
julia> import Pkg; Pkg.activate(".")
```
(as a shortcut, you can also press the `]` key and type `activate .`).

## Instantiating the environment
The first time you run this, you'll also need install the dependencies and precompile them (you only have to do this once):
```julia-repl
julia> Pkg.instantiate()
```
This will take a little while.

Then you'll be able to run the codes here.

# How to run
Each time you re-open Julia, you'll need to [reactivate the environment](#activating-the-environment) in this folder before running any of the codes.

There are currently two mostly unrelated things here:
  - A parallel-tempered MCMC code in `pt.jl`
  - A few utilities for loading and plotting the sangria data in `sangria.jl`

## Parallel-tempered MCMC

This code is a mostly textbook implementation of a [parallel-tempered MCMC](https://en.wikipedia.org/wiki/Parallel_tempering).
It is based off of a geophysics review paper (link??).

You can run a test of the code (implementing a fit of the data from [Hogg et al (2010)](https://arxiv.org/abs/1008.4686)) with
```julia-repl
julia> include("pt.jl")
julia> demo()
```
This will create the file `corner_pt.png` with a corner plot of the samples.

Some caveats which could affect how you want to use the code, but are fixable:
  -  the number of threads and the number of chains are currently assumed to be identical
  -  the priors are currently assumed to be a product distribution (i.e. all variables' priors are independent)
  -  the proposals are currently hardcoded to a truncated gaussian distribution (stdev in each axis equal to the prior's stdev / 10 )
