#=
```@meta
Description = "Meta-optimisers in PortfolioOptimisers.jl: nested clustered, stacking and subset resampling, which split a problem and recombine the pieces."
```

# Meta-optimisers

Every optimiser so far gives weights by solving one problem. A meta-optimiser runs other
optimisers instead. It splits the problem, solves each piece with the estimator you name, and
puts the pieces back together. One fit over every asset rests on one set of estimates, and a
meta-optimiser spreads that risk. It also lets you apply a different rule to a different part
of the universe.

`PortfolioOptimisers` has three, and each one has an inner slot and an outer slot.

  - [`NestedClustered`](@ref), or NCO, clusters the assets, runs an inner optimiser inside
    each cluster, and runs an outer optimiser over the clusters.
  - [`Stacking`](@ref) runs several inner optimisers over every asset, then combines the
    portfolios they give with one outer optimiser.
  - [`SubsetResampling`](@ref) optimises over random subsets of the assets many times and
    averages the weights, which is bagging applied to a portfolio.

Both slots take any optimisation estimator, including another meta-optimiser, so you can nest
them as deep as you like.

!!! tip "When to reach for this"
    Reach for a meta-optimiser when one fit over every asset rests on more estimates than you
    trust. Use NCO when the cluster structure is sound and you want a different rule inside a
    cluster and across clusters. Use Stacking when you want several rules to share the
    decision rather than one. Use SubsetResampling when you want the weights to depend less on
    which assets are in the universe. If one optimiser already does the job, use it, because a
    meta-optimiser costs more time and more settings.
=#

using PortfolioOptimisers, PrettyTables, StableRNGs

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. ReturnsResult data and shared ingredients

We use the same S&P 500 slice as the other optimiser examples. We build a prior, a
clustering and a solver once, and every meta-optimiser below reads them.
=#

using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
pr = prior(EmpiricalPrior(), rd)
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)

#=
The two slots take their statistics from different places, and the cell below shows how. The
[`MeanRisk` objectives](01_MeanRisk_Objectives.md) page covers the wider point, that a slot
takes a computed result or an estimator that computes one.

The inner optimiser gets the prior we computed above, through `pe = pr` on its
[`JuMPOptimiser`](@ref). It solves over the returns of the real assets, and that is what the
prior describes.

The outer optimiser gets no prior. It solves over the returns the meta-optimiser builds, one
series per cluster for NCO and one per inner portfolio for [`Stacking`](@ref). A prior over the
assets does not describe those series. The outer optimiser computes what it needs from them
when it solves, so it needs only a solver here. Pass `pe = pr` to the outer optimiser and it
reads the asset prior in place of the statistics of those series.
=#

jopti = JuMPOptimiser(; pe = pr, slv = slv)
jopto = JuMPOptimiser(; slv = slv)

#=
We also solve a plain minimum-variance [`MeanRisk`](@ref) over every asset, to compare
against.
=#

res_bench = optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
## 2. Nested clustered optimisation (NCO)

NCO solves a minimum-variance problem inside each cluster. It then turns each cluster into
one series of returns and solves a second minimum-variance problem over those series. The two
slots are independent. Both hold a [`MeanRisk`](@ref) here, and either one takes a
risk-budgeting, hierarchical or naive estimator instead.
=#

res_nco = optimise(NestedClustered(; pe = pr, cle = clr,
                                   opti = MeanRisk(; obj = MinimumRisk(), opt = jopti),
                                   opto = MeanRisk(; obj = MinimumRisk(), opt = jopto)), rd)

#=
## 3. Stacking

Stacking runs a list of inner optimisers over every asset. The list below holds a
minimum-variance [`MeanRisk`](@ref), a [`HierarchicalRiskParity`](@ref) and a naive
[`InverseVolatility`](@ref). The outer optimiser then combines the three portfolios they give.
No single rule sets the result on its own.
=#

res_stk = optimise(Stacking(; pe = pr,
                            opti = [MeanRisk(; opt = jopti),
                                    HierarchicalRiskParity(;
                                                           opt = HierarchicalOptimiser(;
                                                                                       pe = pr)),
                                    InverseVolatility(; pe = pr)],
                            opto = MeanRisk(; obj = MinimumRisk(), opt = jopto)), rd)

#=
## 4. Subset resampling

SubsetResampling draws a random subset of the assets, optimises over it, and repeats. It then
averages the weights it collected. We draw 10 subsets, each holding 70% of the assets. The
random number generator and the seed are fixed, so the cell gives the same weights on every
run.
=#

res_ssr = optimise(SubsetResampling(; pe = pr,
                                    opt = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; slv = slv)),
                                    subset_size = 0.7, n_subsets = 10, rng = StableRNG(123),
                                    seed = 42), rd)

#=
## 5. Comparing the allocations

All four columns below target minimum variance and get there by different routes. Read the
three meta-optimiser columns against the plain fit, and read how far each one spreads the
weight over the assets.
=#

pretty_table(DataFrame(; :assets => rd.nx, :MinVar => res_bench.w, :NCO => res_nco.w,
                       :Stacking => res_stk.w, :SubsetResampling => res_ssr.w);
             formatters = [resfmt])

#=
## 6. Visualising the compositions

The plot stacks the same four allocations, with the plain minimum-variance fit first.
=#

# Composition of the benchmark and the three meta-optimisers.
using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_bench, res_nco, res_stk, res_ssr], rd)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - All three meta-optimisers solve to OptimisationSuccess on the docs slice with the
#src   verified inner/outer composition (inner JuMPOptimiser carries `pe = pr`, outer carries
#src   only `slv`; meta-level `pe`/`cle` provided, optimise called with `rd`).
#src - ERGO (record-only per hybrid policy → issue #125): the inner-has-prior / outer-no-prior
#src   convention is load-bearing but only discoverable from the tests — passing `pe = pr` to
#src   the *outer* JuMPOptimiser silently double-counts the prior. Worth a sentence in the
#src   NestedClustered/Stacking docstrings stating that the outer optimiser consumes synthetic
#src   aggregate returns and should not be given the asset-level prior.
