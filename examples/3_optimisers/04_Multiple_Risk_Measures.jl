#=
```@meta
Description = "Multiple risk measures in one PortfolioOptimisers.jl optimisation: combine them with a scalariser, and report a portfolio's risk under all of them."
```

# Multiple risk measures

This page puts several risk measures into one optimisation. It combines them in the objective of
`MeanRisk` and of a hierarchical optimiser, and it reports the risk of a portfolio under a vector
of risk measures.

!!! tip "When to reach for this"
    Reach for several risk measures when one measure does not cover what you care about, and
    you want one optimisation to take *all of them* into account. For example, you can bound
    the variance and also limit a tail measure such as CVaR or a drawdown. A
    [`MeanRisk`](@ref) optimisation takes a vector of risk measures, each with its own
    settings, so the objective and the constraints can combine them. If you want to trade
    several criteria against each other over many portfolios instead, see the
    [Pareto surface](03_Pareto_Surface.md) example.

!!! note "The return side takes several terms too"
    The `ret` field of `JuMPOptimiser` takes one return term or a vector of them, as the `r`
    field of `MeanRisk` does, each with its own [`JuMPReturnsSettings`](@ref). The optimiser
    combines several risk measures with a scalariser, the rule that turns their values into one
    number, and you choose it. It always adds the return terms as a weighted sum, with no
    scalariser. See [ℓ1 uncertainty sets](../2_moments_priors/11_L1_Uncertainty_Quintile_Portfolios.md).
=#
using PortfolioOptimisers, PrettyTables
tsfmt = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Data

We load the same prices as the previous examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

rd = prices_to_returns(X)

#=
## 2. Solvers

We pass a vector of four solvers. When one fails to converge, the optimiser tries the next.
=#

using Clarabel
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.8),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.70),
              check_sol = (; allow_local = true, allow_almost = true))];

#=
## 3. Multiple risk measures

### 3.1 Equally weighted sum

Some risk measures can hold their own prior statistics, and the optimisation uses those instead
of the ones in the prior result. We use this to minimise the variance under several covariance
matrices at once.

We compute the prior once, and define five covariance estimators.

  1. The default covariance, denoised with the spectral algorithm.
  2. The Gerber covariance, whose default algorithm is `Gerber1`.
  3. The Smyth-Broby covariance with the `SmythBroby1` algorithm.
  4. The mutual information covariance.
  5. The distance covariance.
=#
pr = prior(HighOrderPriorEstimator(), rd.X)

ces = [PortfolioOptimisersCovariance(;
                                     mp = MatrixProcessing(;
                                                           dn = Denoise(;
                                                                        alg = SpectralDenoise()))),
       PortfolioOptimisersCovariance(; ce = GerberCovariance()),
       PortfolioOptimisersCovariance(; ce = SmythBrobyCovariance(; alg = SmythBroby1())),
       PortfolioOptimisersCovariance(; ce = MutualInfoCovariance()),
       PortfolioOptimisersCovariance(; ce = DistanceCovariance())]
#=
We build one [`Variance`](@ref) from each covariance matrix, and a sixth from the sum of the five
matrices.
=#
rs = [Variance(; sigma = cov(ce, rd.X)) for ce in ces]
all_sigmas = zeros(length(rd.nx), length(rd.nx))
for r in rs
    all_sigmas .+= r.sigma
end
push!(rs, Variance(; sigma = all_sigmas))

#=
We minimise each of the six variances alone, and average the weights of the first five
portfolios. Then we minimise all six variances in one optimisation, whose default scalariser adds
them.

That sum is twice the variance under the summed covariance, so the `multi_risk` column shows the
same portfolio as the `sum_covs` column, up to the tolerance of the solver. The `mean_w` column
averages the weights of the five single portfolios, and that is a different portfolio.
=#
results = [optimise(MeanRisk(; r = r, opt = JuMPOptimiser(; pe = pr, slv = slv)))
           for r in rs]
mean_w = zeros(length(results[1].w))
for res in results[1:5]
    mean_w .+= res.w
end
mean_w ./= 5
res = optimise(MeanRisk(; r = rs, opt = JuMPOptimiser(; pe = pr, slv = slv)))
pretty_table(DataFrame(:assets => rd.nx, :dn => results[1].w, :gerber1 => results[2].w,
                       :smyth_broby1 => results[3].w, :mutual_info => results[4].w,
                       :distance => results[5].w, :mean_w => mean_w,
                       :sum_covs => results[6].w, :multi_risk => res.w);
             formatters = [resfmt])
#=
We repeat the comparison with the objective that maximises the risk-adjusted return ratio.
=#
results = [optimise(MeanRisk(; r = r, obj = MaximumRatio(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv))) for r in rs]
mean_w = zeros(length(results[1].w))
for res in results[1:5]
    mean_w .+= res.w
end
mean_w ./= 5
res = optimise(MeanRisk(; r = rs, obj = MaximumRatio(),
                        opt = JuMPOptimiser(; pe = pr, slv = slv)))

pretty_table(DataFrame(:assets => rd.nx, :dn => results[1].w, :gerber1 => results[2].w,
                       :smyth_broby1 => results[3].w, :mutual_info => results[4].w,
                       :distance => results[5].w, :mean_w => mean_w,
                       :sum_covs => results[6].w, :multi_risk => res.w);
             formatters = [resfmt])
#=
### 3.2 Different weights and scalarisers

`MeanRisk`, [`NearOptimalCentering`](@ref), [`RiskBudgeting`](@ref),
[`FactorRiskContribution`](@ref), [`HierarchicalRiskParity`](@ref) and
[`HierarchicalEqualRiskContribution`](@ref) each take a vector of risk measures. You can give
each measure its own weight, and choose one of four scalarisers to combine them.
[`SumScalariser`](@ref), [`MaxScalariser`](@ref) and [`LogSumExpScalariser`](@ref) work with all
six, and [`MinScalariser`](@ref) works only with the two hierarchical optimisers.

The restriction on `MinScalariser` comes from the optimisation *model*. Minimising a minimum has
no convex form, so a `JuMP` optimiser does not accept `MinScalariser`. When you only combine risk
values you already computed, no model is built, and all four scalarisers work. Section 5 reports
one portfolio's risk under each of them.

In a hierarchical optimiser the scalariser applies to each sub-optimisation separately. Under
`MaxScalariser`, for example, the larger measure in one cluster need not be the larger measure in
another cluster, or in the whole portfolio. This cannot be avoided. A hierarchical optimiser aims
at a trade-off between risk and diversification, not at the lowest possible risk.

You can mix any risk measures that the optimiser accepts. We mix the variance with the negative
skewness, and we scale the negative skewness down so that the larger of the two measures differs
from cluster to cluster.

We use [`HierarchicalEqualRiskContribution`](@ref), which takes a risk measure and a scalariser
inside each cluster, `ri` and `scai`, and a pair between the clusters, `ro` and `scao`. We compute
the clusters once in advance, with the direct bubble hierarchical tree algorithm,
[`DBHT`](@ref).
=#
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)

#=
## 4. Hierarchical optimisation with two risk measures

Before we optimise, we plot the clusters that the estimator found from the correlation matrix.
=#

using StatsPlots, GraphRecipes
# The dendrogram shows the hierarchical clustering of the assets.
plot_dendrogram(clr, rd.nx)
# The heatmap shows the correlation matrix in the order of the clusters, with a box around each
# cluster.
plot_clusters(clr, rd.nx)

#=
We run six optimisations: the variance alone, the negative skewness alone, and both measures
under each of the four scalarisers. The negative skewness has a scale of 0.1.
=#

r = [Variance(), NegativeSkewness(; settings = RiskMeasureSettings(; scale = 0.1))]

results = [optimise(HierarchicalEqualRiskContribution(; ri = r[1], ro = r[1],
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r[2], ro = r[2],
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = SumScalariser(),
                                                      scao = SumScalariser(),
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = MaxScalariser(),
                                                      scao = MaxScalariser(),
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = MinScalariser(),
                                                      scao = MinScalariser(),
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = LogSumExpScalariser(),
                                                      scao = LogSumExpScalariser(),
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr)))]

pretty_table(DataFrame(:assets => rd.nx, :variance => results[1].w,
                       :neg_skew => results[2].w, :sum_sca => results[3].w,
                       :max_sca => results[4].w, :min_sca => results[5].w,
                       :log_sum_exp => results[6].w); formatters = [resfmt])

#=
The stacked bars show the six portfolios, with the negative skewness at a scale of 0.1.
=#

plot_stacked_bar_composition(results, rd)

#=
We run the same six optimisations with both measures at a scale of 1. If one measure is larger
than the other in every cluster, `MaxScalariser` and `MinScalariser` each give the portfolio of a
single measure.
=#

r = [Variance(), NegativeSkewness()]

results = [optimise(HierarchicalEqualRiskContribution(; ri = r[1], ro = r[1],
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r[2], ro = r[2],
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = SumScalariser(),
                                                      scao = SumScalariser(),
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = MaxScalariser(),
                                                      scao = MaxScalariser(),
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = MinScalariser(),
                                                      scao = MinScalariser(),
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = LogSumExpScalariser(),
                                                      scao = LogSumExpScalariser(),
                                                      opt = HierarchicalOptimiser(; pe = pr,
                                                                                  cle = clr)))]

pretty_table(DataFrame(:assets => rd.nx, :variance => results[1].w,
                       :neg_skew => results[2].w, :sum_sca => results[3].w,
                       :max_sca => results[4].w, :min_sca => results[5].w,
                       :log_sum_exp => results[6].w); formatters = [resfmt])

#=
The stacked bars show the six portfolios with both measures at a scale of 1.
=#

plot_stacked_bar_composition(results, rd)

#=
The `MaxScalariser` portfolio has the same weights as the negative skewness portfolio, and the
`MinScalariser` portfolio the same weights as the variance portfolio. In every cluster the
negative skewness is larger than the variance, so the maximum always picks the negative skewness
and the minimum always picks the variance. [`HierarchicalRiskParity`](@ref) applies its
scalariser at each split of the tree, so the same effect can happen there.
[`NearOptimalCentering`](@ref) computes the risk bounds of its frontier with the scalariser, and
`MaxScalariser` takes the maximum at each end of the frontier separately. A different measure can
then set each bound.

## 5. Reporting a vector of risk measures

[`expected_risk`](@ref) also takes a vector of risk measures. It combines them into *one* number
with the scalariser, and it does not return one number per measure.

`res.r` is the vector of fitted risk measures of the optimisation, and `res.sca` is its
scalariser. If you pass both from the result, the reported risk is the risk that the
optimisation measured.

Here `res` is the maximum-ratio optimisation over all six variance measures from section 3.1.
=#

rk_opt = expected_risk(res.r, res.w, res.pr; sca = res.sca)

#=
We also name the measures and the scalariser by hand, and print both numbers. We name the same
vector and the same scalariser, so the two numbers are equal. A different vector, scalariser or
`fees` gives a different number.
=#

rk_hand = expected_risk(rs, res.w, res.pr; sca = SumScalariser())
pretty_table(DataFrame(; :route => ["from the result", "named by hand"],
                       :risk => [rk_opt, rk_hand]);
             title = "Expected risk of the maximum-ratio portfolio")

#=
The scalariser is a keyword of `expected_risk`, so we report the same portfolio under each of the
four.
=#

scas = [SumScalariser(), MaxScalariser(), MinScalariser(), LogSumExpScalariser()]
pretty_table(DataFrame(; :scalariser => ["Sum", "Max", "Min", "LogSumExp"],
                       :risk =>
                           [expected_risk(res.r, res.w, res.pr; sca = s) for s in scas]);
             title = "Risk of the same portfolio under each scalariser")

#=
Sum, Max and Min stay in the units of the measure, and they always satisfy `Min ≤ Max ≤ Sum`.
Each scalariser first multiplies each risk by its own `settings.scale`. Sum then adds all six
positive values, and Max and Min each keep one of them.

[`LogSumExpScalariser`](@ref) gives a much larger number than the other three. It is a smooth
maximum computed in log space, `log(Σᵢ exp(γ rᵢ)) / γ` with the default `γ = 1`, so it is not a
weighted average of the risks
and it is not in their units. When the risks are small, `log(N)`, where `N` is the number of
measures, makes up most of its value. Use it as a smooth form of the maximum, and do not compare
its value with the other three.

A hierarchical result also stores its measures, one pair for each level. The result of
`HierarchicalEqualRiskContribution` contains the fitted `ri`, `scai`, `ro` and `scao`. We report the
risk of the `SumScalariser` portfolio from the second run of section 4 under its outer measures
and scalariser, `ro` and `scao`.
=#

res_herc = results[3]
rk_herc = expected_risk(res_herc.ro, res_herc.w, res_herc.pr; sca = res_herc.scao)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end (6 covariance estimators incl. MutualInfo/Distance, equal-weight
#src   + scalariser HERC sweeps, dendrogram/cluster/composition plots). Results match the
#src   narrative (Max collapses to NegativeSkewness-only, Min to Variance-only).
#src - PLOTTING DEPRECATION (record-only per hybrid policy → issue #125): `plot_clusters`
#src   emits `Warning: Keyword argument `orientation` is deprecated. Please use `permute`
#src   instead.` Origin: ext/PortfolioOptimisersPlotsExt.jl:468, the sideways `dend2`
#src   (`orientation = :horizontal`). Likely fix `permute = (:x, :y)`, but it interacts with
#src   the `xlim = extrema(heights)`/`yflip`/`xrotation` on the same call, so it needs a
#src   rendered visual check before changing — not fixed blind here.
