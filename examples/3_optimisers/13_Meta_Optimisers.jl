#=
```@meta
Description = "Meta-optimisers in PortfolioOptimisers.jl: nested clustered, stacking and subset resampling, which split a problem and recombine the pieces."
```

# [Meta-optimisers](@id example-meta-optimisers)

Every optimiser so far gives weights by solving one problem. A meta-optimiser runs other
optimisers instead. It splits the problem, solves each piece with the estimator you name, and
puts the pieces back together. One fit over every asset depends on one set of estimates, and
a meta-optimiser splits that dependence across several fits. It also lets you apply a different
rule to a different part of the universe.

`PortfolioOptimisers` has three.

  - [`NestedClustered`](@ref), or NCO, clusters the assets, runs an inner optimiser inside
    each cluster, and runs an outer optimiser over the clusters.
  - [`Stacking`](@ref) runs several inner optimisers over every asset, then combines the
    portfolios they give with one outer optimiser.
  - [`SubsetResampling`](@ref) optimises over random subsets of the assets many times and
    averages the weights, which is bagging applied to a portfolio.

`NestedClustered` takes one inner optimiser in `opti` and an outer one in `opto`. `Stacking`
takes a list of inner optimisers in `opti` and an outer one in `opto`. `SubsetResampling` takes
one optimiser in `opt`. Each of these keywords takes any optimiser that gives continuous weights,
including another meta-optimiser, so you can nest them as deep as you like.

!!! tip "When to reach for this"
    Reach for a meta-optimiser when one fit over every asset depends on more estimates than you
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
## 1. Data and shared ingredients

We use one year of daily prices for twenty S&P 500 stocks. We build a prior, a clustering and
a solver once, and every meta-optimiser below uses them.
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
The inner and the outer optimiser take their statistics from different places, and the cell
below shows how. The [`MeanRisk` objectives](@ref example-meanrisk-objectives) page covers the wider
point, that a keyword takes a computed result or an estimator that computes one.

The inner optimiser gets the prior we computed above, through `pe = pr` on its
[`JuMPOptimiser`](@ref). It solves over the returns of the real assets, and that is what the
prior describes.

The outer optimiser gets no prior. It solves over the returns the meta-optimiser builds, one
series per cluster for NCO and one per inner portfolio for [`Stacking`](@ref). A prior over the
assets does not describe those series. The outer optimiser computes what it needs from them
when it solves, so it needs only a solver here. The constructor refuses a computed prior on
the outer optimiser.
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
one series of returns and solves a second minimum-variance problem over those series. The
inner and the outer optimiser are independent. Both hold a [`MeanRisk`](@ref) here, and either one takes a
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

The plain fit, NCO and subset resampling target minimum variance. The stacked portfolio
combines three inner portfolios by minimum variance, and only one of them, the `MeanRisk` fit,
targets minimum variance itself. The outer fit measures the variance of each mix on the same
rows and with the same covariance estimator as that inner fit, and every long-only mix of the
three members is a portfolio the inner fit could have chosen. No mix has a lower variance, so
on any returns the outer fit puts all its weight on the minimum-variance member, and the
`Stacking` column is the `MinVar` column. Read the other two meta-optimiser columns against
the plain fit, and read how far each one spreads the weight over the assets.
=#

pretty_table(DataFrame(; :assets => rd.nx, :MinVar => res_bench.w, :NCO => res_nco.w,
                       :Stacking => res_stk.w, :SubsetResampling => res_ssr.w);
             formatters = [resfmt])

#=
## 6. Visualising the compositions

The plot stacks the same four allocations, with the plain minimum-variance fit first.
=#

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
