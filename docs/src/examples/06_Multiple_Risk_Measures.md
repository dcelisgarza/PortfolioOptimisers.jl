The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/06_Multiple_Risk_Measures.jl"
```

# Example 6: Multiple risk measures

This example shows how to use multiple risk measures.

````@example 06_Multiple_Risk_Measures
using PortfolioOptimisers, PrettyTables
# Format for pretty tables.
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
mipresfmt = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. ReturnsResult data

We will use the same data as the previous example.

````@example 06_Multiple_Risk_Measures
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X)
````

## 2. Preparatory steps

We'll provide a vector of continuous solvers as a failsafe.

````@example 06_Multiple_Risk_Measures
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
nothing #hide
````

## 3. Multiple risk measures

### 3.1 Equally weighted sum

Some risk measures can use precomputed prior statistics which take precedence over the ones in `PriorResult`. We can make use of this to minimise the variance with different covariance matrices simultaneously.

We will also precompute the prior statistics to minimise redundant work. First lets create a vector of Variances onto which we will push the different variances. We'll use 5 variance estimators, and their equally weighted sum.

  1. Denoised covariance using the spectral algorithm.
  2. Gerber 1 covariance.
  3. Smyth Broby 1 covariance.
  4. Mutual Information covariance.
  5. Distance covariance.
  6. Equally weighted sum of all the above covariances.

For the multi risk measure optimisation, we will weigh each risk measure equally. It should give the same result as adding all covariances together, but not the same as averaging the weights of the individual optimisations.

````@example 06_Multiple_Risk_Measures
pr = prior(HighOrderPriorEstimator(), rd.X)

ces = [PortfolioOptimisersCovariance(;
                                     mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                           dn = Denoise(;
                                                                                        alg = SpectralDenoise()))),
       PortfolioOptimisersCovariance(; ce = GerberCovariance()),
       PortfolioOptimisersCovariance(; ce = SmythBrobyCovariance(; alg = SmythBroby1())),
       PortfolioOptimisersCovariance(; ce = MutualInfoCovariance()),
       PortfolioOptimisersCovariance(; ce = DistanceCovariance())]
````

Lets define a vector of variance risk measure using each of the different covariance matrices.

````@example 06_Multiple_Risk_Measures
rs = [Variance(; sigma = cov(ce, rd.X)) for ce in ces]
all_sigmas = zeros(length(rd.nx), length(rd.nx))
for r in rs
    all_sigmas .+= r.sigma
end
push!(rs, Variance(; sigma = all_sigmas))
````

We'll minimise the variance for each individual risk measure and then we'll minimise the equally weighted sum of all risk measures.

````@example 06_Multiple_Risk_Measures
results = [optimise(MeanRisk(; r = r, opt = JuMPOptimiser(; pr = pr, slv = slv)))
           for r in rs]
mean_w = zeros(length(results[1].w))
for res in results[1:5]
    mean_w .+= res.w
end
mean_w ./= 5
res = optimise(MeanRisk(; r = rs, opt = JuMPOptimiser(; pr = pr, slv = slv)))
pretty_table(DataFrame(:assets => rd.nx, :denoise => results[1].w, :gerber1 => results[2].w,
                       :smyth_broby1 => results[3].w, :mutual_info => results[4].w,
                       :distance => results[5].w, :mean_w => mean_w,
                       :sum_covs => results[6].w, :multi_risk => res.w);
             formatters = [resfmt])
````

For extra credit we can do the same but maximising the risk-adjusted return ratio.

````@example 06_Multiple_Risk_Measures
results = [optimise(MeanRisk(; r = r, obj = MaximumRatio(),
                             opt = JuMPOptimiser(; pr = pr, slv = slv))) for r in rs]
mean_w = zeros(length(results[1].w))
for res in results[1:5]
    mean_w .+= res.w
end
mean_w ./= 5
res = optimise(MeanRisk(; r = rs, obj = MaximumRatio(),
                        opt = JuMPOptimiser(; pr = pr, slv = slv)))

pretty_table(DataFrame(:assets => rd.nx, :denoise => results[1].w, :gerber1 => results[2].w,
                       :smyth_broby1 => results[3].w, :mutual_info => results[4].w,
                       :distance => results[5].w, :mean_w => mean_w,
                       :sum_covs => results[6].w, :multi_risk => res.w);
             formatters = [resfmt])
````

### 3.2 Different weights and scalarisers

All optimisations accept multiple risk measures in the same way. We can also provide different weights for each measure and four different scalarisers, [`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`LogSumExpScalariser`](@ref) which work for all optimisation estimators, and [`MinScalariser`](@ref) which only works for hierarchical ones.

For clustering optimisations, the scalarisers apply to each sub-optimisation, so what may be the choice of risk to "minimise" for one cluster may not be the minimal risk for others, or the overall portfolio. This inconsistency is unavoidable but should not be a problem in practice as the point of hierarchical optimisations is not to provide the absolute minimum risk, but a good trade-off between risk and diversification.

It is also possible to mix any and all compatible risk measures. We will demonstrate this by mixing the variance with the negative skewness.

In this example we have tuned the weight of the negative skewness to demonstrate how clusters may end up with different risk measures due to the choice of scalariser.

We will use the heirarchical equal risk contribution optimisation, precomputing the clustering results using the direct bubble hierarchy tree algorithm.

The [`HierarchicalEqualRiskContribution`]-(@ref) optimisation estimator accepts inner and outer risk measures and inner and outer scalarisers.

````@example 06_Multiple_Risk_Measures
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)
r = [Variance(), NegativeSkewness(; settings = RiskMeasureSettings(; scale = 0.1))]

results = [optimise(HierarchicalEqualRiskContribution(; ri = r[1],# inner (intra-cluster) risk measure
                                                      ro = r[1],# outer (inter-cluster) risk measure
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r[2], ro = r[2],
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,#
                                                      scai = SumScalariser(),# inner (intra-cluster)
                                                      scao = SumScalariser(),# outer (inter-cluster)
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = MaxScalariser(),
                                                      scao = MaxScalariser(),
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = MinScalariser(),
                                                      scao = MinScalariser(),
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = LogSumExpScalariser(),
                                                      scao = LogSumExpScalariser(),
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr)))]

pretty_table(DataFrame(:assets => rd.nx, :variance => results[1].w,
                       :neg_skew => results[2].w, :sum_sca => results[3].w,
                       :max_sca => results[4].w, :min_sca => results[5].w,
                       :log_sum_exp => results[6].w); formatters = [resfmt])
````

When the weights are different enough that one risk measure domintes over the other in all contexts, then the results of the max and min scalarisers will be as expected, i.e. they will be as if only one risk measure was used.

````@example 06_Multiple_Risk_Measures
r = [Variance(), NegativeSkewness()]

results = [optimise(HierarchicalEqualRiskContribution(; ri = r[1],# inner (intra-cluster) risk measure
                                                      ro = r[1],# outer (inter-cluster) risk measure
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r[2], ro = r[2],
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,#
                                                      scai = SumScalariser(),# inner (intra-cluster)
                                                      scao = SumScalariser(),# outer (inter-cluster)
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = MaxScalariser(),
                                                      scao = MaxScalariser(),
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = MinScalariser(),
                                                      scao = MinScalariser(),
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr))),
           optimise(HierarchicalEqualRiskContribution(; ri = r, ro = r,
                                                      scai = LogSumExpScalariser(),
                                                      scao = LogSumExpScalariser(),
                                                      opt = HierarchicalOptimiser(; pr = pr,
                                                                                  clr = clr)))]

pretty_table(DataFrame(:assets => rd.nx, :variance => results[1].w,
                       :neg_skew => results[2].w, :sum_sca => results[3].w,
                       :max_sca => results[4].w, :min_sca => results[5].w,
                       :log_sum_exp => results[6].w); formatters = [resfmt])
````

Note how the max scalariser produced the same weights as the negative skewness and the min scalariser produced the same weights as the variance. This is because in all cases, the same the value of the negative skewness was greater than that of the variance. A similar behaviour can be observed with other clustering optimisers. [`NearOptimalCentering`]-(@ref) can also have unintuitive behaviour when computing the risk bounds with an effective frontier [`MaxScalariser`](@ref) and [`MinScalariser`](@ref) due to the fact that each point in the efficient frontier can have a different risk measure dominating the others.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
