# PortfolioOptimisers.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimisers.jl/dev)
[![Test workflow status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl)
[![Docs workflow Status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Build Status](https://api.cirrus-ci.com/github/dcelisgarza/PortfolioOptimisers.jl.svg)](https://cirrus-ci.com/github/dcelisgarza/PortfolioOptimisers.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/dcelisgarza/PortfolioOptimisers.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Welcome to PortfolioOptimisers.jl

 [`PortfolioOptimisers.jl`](https://github.com/dcelisgarza/PortfolioOptimisers.jl) is a package for portfolio optimisation written in Julia.

!!! Danger
    Investing conveys real risk, the entire point of portfolio optimisation is to minimise it to tolerable levels. The examples use outdated data and a variety of stocks (including what I consider to be meme stocks) for demonstration purposes only. None of the information in this documentation should be taken as financial advice. Any advice is limited to improving portfolio construction, most of which is common investment and statistical knowledge.

Portfolio optimisation is the science of either:

- Minimising risk whilst keeping returns to acceptable levels.
- Maximising returns whilst keeping risk to acceptable levels.

To some definition of acceptable, and with any number of additional constraints available to the optimisation type.

There exist myriad statistical, pre- and post-processing, optimisations, and constraints that allow one to explore a vast landscape of "optimal" portfolios.

`PortfolioOptimisers.jl` is an attempt at providing as many of these as possible under a single banner. We make extensive use of `Julia`'s type system, module extensions, and multiple dispatch to simplify development and maintenance.

For more information on the package's *vast* feature list, please check out the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction) and [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_API_Introduction) docs.

## Caveat emptor

- `PortfolioOptimisers.jl` is under active development and still in `v0.*.*`. Therefore, breaking changes should be expected with `v0.X.0` releases. All other releases will fall under `v0.X.Y`.
- The documentation is still under construction.
- Testing coverage is still under `95 %`. We're mainly missing assertion tests, but some lesser used features are partially or wholly untested.
- Please feel free to submit issues, discussions and/or PRs regarding missing docs, examples, features, tests, and bugs.

## Installation

`PortfolioOptimisers.jl` is a registered package, so installation is as simple as:

```julia
julia> using Pkg

julia> Pkg.add(PackageSpec(; name = "PortfolioOptimisers"))
```

## Quick-start

The library is quite powerful and extremely flexible. Here is what a very basic end-to-end workflow can look like. The [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction) contain more thorough explanations and demos. The [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_API_Introduction) docs contain toy examples of the many, many features.

First we import the packages we will need for the example.

- `StatsPlots` and `GraphRecipes` are needed to load the `Plots.jl` extension.
- `Clarabel` and `HiGHS` are the optimisers we will use.
- `YFinance` and `TimeSeries` for downloading and preprocessing price data.
- `PrettyTables` and `DataFrames` for displaying the results.

```julia
# Import module and plotting extension.
using PortfolioOptimisers, StatsPlots, GraphRecipes
# Import optimisers.
using Clarabel, HiGHS
# Download data and pretty printing
using YFinance, PrettyTables, TimeSeries, DataFrames

# Format for pretty tables.
fmt1 = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end

fmt2 = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end

# Function to convert prices to time array.
function stock_price_to_time_array(x)
    # Only get the keys that are not ticker or datetime.
    coln = collect(keys(x))[3:end]
    # Convert the dictionary into a matrix.
    m = hcat([x[k] for k in coln]...)
    return TimeArray(x["timestamp"], m, Symbol.(coln), x["ticker"])
end

# Tickers to download. These are popular meme stocks, use something better.
assets = sort!(["SOUN", "RIVN", "GME", "AMC", "SOFI", "ENVX", "ANVS", "LUNR", "EOSE", "SMR",
                "NVAX", "UPST", "ACHR", "RKLB", "MARA", "LGVN", "LCID", "CHPT", "MAXN",
                "BB"])

# Prices date range.
Date_0 = "2024-01-01"
Date_1 = "2025-10-05"

# Download the price data using YFinance.
prices = get_prices.(assets; startdt = Date_0, enddt = Date_1)
prices = stock_price_to_time_array.(prices)
prices = hcat(prices...)
cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
prices = prices[cidx]
TimeSeries.rename!(prices, Symbol.(assets))
pretty_table(prices[(end - 5):end]; formatters = [fmt1])
#=
┌────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│  timestamp │    ACHR │     AMC │    ANVS │      BB │    CHPT │    ENVX │    EOSE │     GME │    LCID │    LGVN │    LUNR │    MARA │    MAXN │    NVAX │    RIVN │    RKLB │     SMR │    SOFI │    SOUN │    UPST │
│   DateTime │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │
├────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ 2025-09-26 │    9.28 │    2.89 │    1.97 │    4.96 │   10.84 │   10.09 │   10.12 │   26.42 │   23.96 │   0.799 │   10.08 │   16.13 │    3.53 │    8.55 │   15.59 │   46.26 │    38.0 │   27.98 │   15.94 │   57.35 │
│ 2025-09-29 │    9.65 │     3.0 │    2.04 │     5.0 │   11.05 │    9.97 │   11.17 │   27.21 │   24.11 │    0.77 │   10.26 │   18.66 │    3.55 │    8.57 │   15.25 │   47.01 │   38.16 │   27.55 │   15.68 │   52.74 │
│ 2025-09-30 │    9.58 │     2.9 │    2.07 │    4.88 │   10.92 │    9.97 │   11.39 │   27.28 │   23.79 │    0.75 │   10.52 │   18.26 │    3.35 │    8.67 │   14.68 │   47.91 │    36.0 │   26.42 │   16.08 │    50.8 │
│ 2025-10-01 │    9.81 │    2.95 │    2.13 │    4.79 │   11.63 │   11.11 │   12.37 │   27.69 │  24.295 │   0.742 │   10.61 │   18.61 │    3.58 │     9.5 │   14.61 │   47.97 │   36.61 │   25.76 │   16.15 │   52.13 │
│ 2025-10-02 │   10.18 │    3.15 │    2.23 │    4.75 │   11.32 │   11.65 │   12.36 │   27.22 │    24.1 │   0.765 │   11.22 │   18.79 │    3.78 │    9.55 │   13.53 │   52.47 │   39.51 │   25.97 │   17.84 │   52.88 │
│ 2025-10-03 │   11.57 │    3.06 │    2.22 │     4.5 │   11.94 │   11.92 │    12.6 │   25.38 │   24.77 │   0.788 │   11.44 │   18.82 │     3.6 │    9.46 │   13.65 │   56.16 │   40.12 │   25.24 │   17.85 │   51.96 │
└────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
=#

# Compute the returns.
rd = prices_to_returns(prices)
#=
ReturnsResult
    nx ┼ 20-element Vector{String}
     X ┼ 440×20 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    ts ┼ 440-element Vector{DateTime}
    iv ┼ nothing
  ivpa ┴ nothing
=#

# Define the continuous solver.
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
             check_sol = (; allow_local = true, allow_almost = true))
#=
Solver
         name ┼ Symbol: :clarabel1
       solver ┼ UnionAll: Clarabel.MOIwrapper.Optimizer
     settings ┼ Dict{String, Real}: Dict{String, Real}("verbose" => false, "max_step_fraction" => 0.9)
    check_sol ┼ @NamedTuple{allow_local::Bool, allow_almost::Bool}: (allow_local = true, allow_almost = true)
  add_bridges ┴ Bool: true
=#

# `PortfolioOptimisers.jl` implements a number of optimisation types as estimators. All the ones which use mathematical optimisation require a `JuMPOptimiser` structure which defines general solver constraints. This structure in turn requires an instance (or vector) of `Solver`.
opt = JuMPOptimiser(; slv = slv);

# Vanilla (Markowitz) mean risk optimisation, i.e. minimum variance portfolio
mr = MeanRisk(; opt = opt)
#=
MeanRisk
  opt ┼ JuMPOptimiser
      │       pr ┼ EmpiricalPrior
      │          │        ce ┼ PortfolioOptimisersCovariance
      │          │           │   ce ┼ Covariance
      │          │           │      │    me ┼ SimpleExpectedReturns
      │          │           │      │       │   w ┴ nothing
      │          │           │      │    ce ┼ GeneralCovariance
      │          │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │          │           │      │       │    w ┴ nothing
      │          │           │      │   alg ┴ Full()
      │          │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
      │          │           │      │     pdm ┼ Posdef
      │          │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │          │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │          │           │      │      dn ┼ nothing
      │          │           │      │      dt ┼ nothing
      │          │           │      │     alg ┼ nothing
      │          │           │      │   order ┴ DenoiseDetoneAlg()
      │          │        me ┼ SimpleExpectedReturns
      │          │           │   w ┴ nothing
      │          │   horizon ┴ nothing
      │      slv ┼ Solver
      │          │          name ┼ Symbol: :clarabel1
      │          │        solver ┼ UnionAll: Clarabel.MOIwrapper.Optimizer
      │          │      settings ┼ Dict{String, Real}: Dict{String, Real}("verbose" => false, "max_step_fraction" => 0.9)
      │          │     check_sol ┼ @NamedTuple{allow_local::Bool, allow_almost::Bool}: (allow_local = true, allow_almost = true)
      │          │   add_bridges ┴ Bool: true
      │       wb ┼ WeightBounds
      │          │   lb ┼ Float64: 0.0
      │          │   ub ┴ Float64: 1.0
      │      bgt ┼ Float64: 1.0
      │     sbgt ┼ nothing
      │       lt ┼ nothing
      │       st ┼ nothing
      │      lcs ┼ nothing
      │       ct ┼ nothing
      │    gcard ┼ nothing
      │   sgcard ┼ nothing
      │     smtx ┼ nothing
      │    sgmtx ┼ nothing
      │      slt ┼ nothing
      │      sst ┼ nothing
      │     sglt ┼ nothing
      │     sgst ┼ nothing
      │       tn ┼ nothing
      │     fees ┼ nothing
      │     sets ┼ nothing
      │       tr ┼ nothing
      │       pl ┼ nothing
      │      ret ┼ ArithmeticReturn
      │          │   ucs ┼ nothing
      │          │    lb ┼ nothing
      │          │    mu ┴ nothing
      │      sca ┼ SumScalariser()
      │     ccnt ┼ nothing
      │     cobj ┼ nothing
      │       sc ┼ Int64: 1
      │       so ┼ Int64: 1
      │       ss ┼ nothing
      │     card ┼ nothing
      │    scard ┼ nothing
      │      nea ┼ nothing
      │       l1 ┼ nothing
      │       l2 ┼ nothing
      │   strict ┴ Bool: false
    r ┼ Variance
      │   settings ┼ RiskMeasureSettings
      │            │   scale ┼ Float64: 1.0
      │            │      ub ┼ nothing
      │            │     rke ┴ Bool: true
      │      sigma ┼ nothing
      │       chol ┼ nothing
      │         rc ┼ nothing
      │        alg ┴ SquaredSOCRiskExpr()
  obj ┼ MinimumRisk()
   wi ┼ nothing
   fb ┴ nothing
=#

# Perform the optimisation, res.w contains the optimal weights.
res = optimise(mr, rd)
#=
MeanRiskResult
       oe ┼ DataType: DataType
       pa ┼ ProcessedJuMPOptimiserAttributes
          │       pr ┼ LowOrderPrior
          │          │         X ┼ 440×20 Matrix{Float64}
          │          │        mu ┼ 20-element Vector{Float64}
          │          │     sigma ┼ 20×20 Matrix{Float64}
          │          │      chol ┼ nothing
          │          │         w ┼ nothing
          │          │       ens ┼ nothing
          │          │       kld ┼ nothing
          │          │        ow ┼ nothing
          │          │        rr ┼ nothing
          │          │      f_mu ┼ nothing
          │          │   f_sigma ┼ nothing
          │          │       f_w ┴ nothing
          │       wb ┼ WeightBounds
          │          │   lb ┼ 20-element StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
          │          │   ub ┴ 20-element StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
          │       lt ┼ nothing
          │       st ┼ nothing
          │      lcs ┼ nothing
          │       ct ┼ nothing
          │    gcard ┼ nothing
          │   sgcard ┼ nothing
          │     smtx ┼ nothing
          │    sgmtx ┼ nothing
          │      slt ┼ nothing
          │      sst ┼ nothing
          │     sglt ┼ nothing
          │     sgst ┼ nothing
          │       tn ┼ nothing
          │     fees ┼ nothing
          │       pl ┼ nothing
          │      ret ┼ ArithmeticReturn
          │          │   ucs ┼ nothing
          │          │    lb ┼ nothing
          │          │    mu ┴ nothing
  retcode ┼ OptimisationSuccess
          │   res ┴ Dict{Any, Any}: Dict{Any, Any}()
      sol ┼ JuMPOptimisationSolution
          │   w ┴ 20-element Vector{Float64}
    model ┼ A JuMP Model
          │ ├ solver: Clarabel
          │ ├ objective_sense: MIN_SENSE
          │ │ └ objective_function_type: JuMP.QuadExpr
          │ ├ num_variables: 21
          │ ├ num_constraints: 4
          │ │ ├ JuMP.AffExpr in MOI.EqualTo{Float64}: 1
          │ │ ├ Vector{JuMP.AffExpr} in MOI.Nonnegatives: 1
          │ │ ├ Vector{JuMP.AffExpr} in MOI.Nonpositives: 1
          │ │ └ Vector{JuMP.AffExpr} in MOI.SecondOrderCone: 1
          │ └ Names registered in the model
          │   └ :G, :bgt, :dev_1, :dev_1_soc, :k, :lw, :obj_expr, :ret, :risk, :risk_vec, :sc, :so, :variance_flag, :variance_risk_1, :w, :w_lb, :w_ub
       fb ┴ nothing
=#

# Define the MIP solver for finite discrete allocation.
mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true));

# Discrete finite allocation.
da = DiscreteAllocation(; slv = mip_slv)
#=
DiscreteAllocation
  slv ┼ Solver
      │          name ┼ Symbol: :highs1
      │        solver ┼ DataType: DataType
      │      settings ┼ Dict{String, Bool}: Dict{String, Bool}("log_to_console" => 0)
      │     check_sol ┼ @NamedTuple{allow_local::Bool, allow_almost::Bool}: (allow_local = true, allow_almost = true)
      │   add_bridges ┴ Bool: true
   sc ┼ Int64: 1
   so ┼ Int64: 1
   wf ┼ AbsoluteErrorWeightFinaliser()
   fb ┼ GreedyAllocation
      │     unit ┼ Int64: 1
      │     args ┼ Tuple{}: ()
      │   kwargs ┼ @NamedTuple{}: NamedTuple()
      │       fb ┴ nothing
=#

# Perform the finite discrete allocation, uses the final asset
# prices, and an available cash amount. This is for us mortals
# without infinite wealth.
mip_res = optimise(da, res.w, vec(values(prices[end])), 4206.90)
#=
DiscreteAllocationResult
         oe ┼ DataType: DataType
    retcode ┼ OptimisationSuccess
            │   res ┴ nothing
  s_retcode ┼ nothing
  l_retcode ┼ OptimisationSuccess
            │   res ┴ Dict{Any, Any}: Dict{Any, Any}()
     shares ┼ 20-element SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
       cost ┼ 20-element SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
          w ┼ 20-element SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
       cash ┼ Float64: 0.2000379800783776
    s_model ┼ nothing
    l_model ┼ A JuMP Model
            │ ├ solver: HiGHS
            │ ├ objective_sense: MIN_SENSE
            │ │ └ objective_function_type: JuMP.AffExpr
            │ ├ num_variables: 21
            │ ├ num_constraints: 42
            │ │ ├ JuMP.AffExpr in MOI.GreaterThan{Float64}: 1
            │ │ ├ Vector{JuMP.AffExpr} in MOI.NormOneCone: 1
            │ │ ├ JuMP.VariableRef in MOI.GreaterThan{Float64}: 20
            │ │ └ JuMP.VariableRef in MOI.Integer: 20
            │ └ Names registered in the model
            │   └ :r, :sc, :so, :u, :x
         fb ┴ nothing
=#

# View the results.
df = DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
               :opt_weights => res.w, :mip_weights => mip_res.w)
pretty_table(df; formatters = [fmt2])
#=
┌────────┬─────────┬─────────┬─────────────┬─────────────┐
│ assets │  shares │    cost │ opt_weights │ mip_weights │
│ String │ Float64 │ Float64 │     Float64 │     Float64 │
├────────┼─────────┼─────────┼─────────────┼─────────────┤
│   ACHR │     0.0 │     0.0 │       0.0 % │       0.0 % │
│    AMC │    73.0 │  223.38 │     5.324 % │      5.31 % │
│   ANVS │    22.0 │   48.84 │     1.249 % │     1.161 % │
│     BB │   273.0 │  1228.5 │    29.184 % │    29.203 % │
│   CHPT │    11.0 │  131.34 │     3.002 % │     3.122 % │
│   ENVX │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   EOSE │     8.0 │   100.8 │     2.435 % │     2.396 % │
│    GME │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   LCID │     1.0 │   24.77 │     0.638 % │     0.589 % │
│   LGVN │   325.0 │   256.1 │     6.089 % │     6.088 % │
│   LUNR │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   MARA │     1.0 │   18.82 │     0.613 % │     0.447 % │
│   MAXN │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   NVAX │    28.0 │  264.88 │      6.21 % │     6.297 % │
│   RIVN │    55.0 │  750.75 │    17.897 % │    17.847 % │
│   RKLB │     4.0 │  224.64 │     4.896 % │      5.34 % │
│    SMR │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   SOFI │    37.0 │  933.88 │    22.462 % │      22.2 % │
│   SOUN │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   UPST │     0.0 │     0.0 │       0.0 % │       0.0 % │
└────────┴─────────┴─────────┴─────────────┴─────────────┘
=#

# Plot the portfolio cumulative returns of the finite allocation portfolio.
plot_ptf_cumulative_returns(mip_res.w, rd.X; ts = rd.ts, compound = true)
```

![Fig. 1](./docs/src/assets/readme_1.svg)

```julia
# Furthermore, we can also plot the risk contribution per asset. For this, we must provide an instance of the risk measure we want to use with the appropriate statistics/parameters. We can do this by using the `factory` function (recommended when doing so programmatically), or manually set the quantities ourselves.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx,
                       percentage = true)

# This awkwardness is due to the fact that `PortfolioOptimisers.jl` tries to decouple the risk measures from optimisation estimators and results. However, the advantage of this approach is that it lets us use multiple different risk measures as part of the risk expression, or as risk limits in optimisations. We explore this further in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction).
```

![Fig. 2](./docs/src/assets/readme_2.svg)

```julia
# We can also plot the returns' histogram and probability density.
plot_histogram(mip_res.w, rd.X, slv)
```

![Fig. 3](./docs/src/assets/readme_3.svg)

```julia
# Plot compounded or uncompounded drawdowns. We use the former here.
plot_drawdowns(mip_res.w, rd.X, slv; ts = rd.ts, compound = true)
```

![Fig. 4](./docs/src/assets/readme_4.svg)

There are other kinds of plots which we explore in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction).

!!! Info
    This section is under active development and any `<name>`-(TBA) lack docstrings. Some docstrings are also outdated, please refer to [Issue #58](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/58) for details on what docstrings have been completed in the `dev` branch.

## Features

### Preprocessing

- Prices to returns [`prices_to_returns`] and [`ReturnsResult`]
- Find complete indices [`find_complete_indices`]
- Find uncorrelated indices [`find_uncorrelated_indices`]-(TBA)

### Matrix Processing

- Positive definite projection [`Posdef`], [`posdef!`], [`posdef`]
- Denoising [`Denoise`], [`denoise!`], [`denoise`]
  - Spectral [`SpectralDenoise`]
  - Fixed [`FixedDenoise`]
  - Shrunk [`ShrunkDenoise`]
- Detoning [`Detone`], [`detone!`], [`detone`]
- Matrix processing pipeline [`DenoiseDetoneAlgMatrixProcessing`], [`matrix_processing!`], [`matrix_processing`], [`DenoiseDetoneAlg`], [`DenoiseAlgDetone`], [`DetoneDenoiseAlg`], [`DetoneAlgDenoise`], [`AlgDenoiseDetone`], [`AlgDetoneDenoise`]

### Regression Models

Factor prior models and implied volatility use [`regression`] in their estimation, which return a [`Regression`] object.

#### Regression targets

- Linear model [`LinearModel`]
- Generalised linear model [`GeneralisedLinearModel`]

#### Regression types

- Stepwise [`StepwiseRegression`]
  - Algorithms
    - Forward [`Forward`]
    - Backward [`Backward`]
  - Selection criteria
    - P-value [`PValue`]
    - Akaike information criteria [`AIC`]
    - Corrected Akaike information criteria [`AICC`]
    - Bayesian information criteria [`BIC`]
    - R-squared [`RSquared`]
    - Adjusted R-squared criteria [`AdjustedRSquared`]

- Dimensional reduction with custom mean and variance estimators [`DimensionReductionRegression`]
  - Dimensional reduction targets
    - Principal component [`PCA`]
    - Probabilistic principal component [`PPCA`]

### Moment Estimation

#### [Expected Returns](@id readme-expected-returns)

Overloads `Statistics.mean`.

- Optionally weighted expected returns [`SimpleExpectedReturns`]
- Equilibrium expected returns with custom covariance [`EquilibriumExpectedReturns`]
- Excess expected returns with custom expected returns estimator [`ExcessExpectedReturns`]
- Shrunk expected returns with custom expected returns and custom covariance estimators [`ShrunkExpectedReturns`]
  - Algorithms
    - James-Stein [`JamesStein`]
    - Bayes-Stein [`BayesStein`]
    - Bodnar-Okhrin-Parolya [`BodnarOkhrinParolya`]
  - Targets: all algorithms can have any of the following targets
    - Grand Mean [`GrandMean`]
    - Volatility Weighted [`VolatilityWeighted`]
    - Mean Squared Error [`MeanSquaredError`]
- Standard deviation expected returns [`StandardDeviationExpectedReturns`]-(TBA)

#### [Variance and Standard Deviation](@id readme-variance)

Overloads `Statistics.var` and `Statistics.std`.

- Optionally weighted variance with custom expected returns estimator [`SimpleVariance`]

#### [Covariance and Correlation](@id readme-covariance-correlation)

Overloads `Statistics.cov` and `Statistics.cor`.

- Optionally weighted covariance with custom covariance estimator [`GeneralCovariance`]
- Covariance with custom covariance estimator [`Covariance`]
  - Full covariance [`Full`]
  - Semi (downside) covariance [`Semi`]
- Gerber covariances with custom variance estimator [`GerberCovariance`]
  - Unstandardised algorithms
    - Gerber 0 [`Gerber0`]
    - Gerber 1 [`Gerber1`]
    - Gerber 2 [`Gerber2`]
  - Standardised algorithms (Z-transforms the data beforehand) with custom expected returns estimator
    - Gerber 0 [`StandardisedGerber0`]
    - Gerber 1 [`StandardisedGerber1`]
    - Gerber 2 [`StandardisedGerber2`]
- Smyth-Broby extension of Gerber covariances with custom expected returns and variance estimators [`SmythBrobyCovariance`]
  - Unstandardised algorithms
    - Smyth-Broby 0 [`SmythBroby0`]
    - Smyth-Broby 1 [`SmythBroby1`]
    - Smyth-Broby 2 [`SmythBroby2`]
    - Smyth-Broby-Gerber 0 [`SmythBrobyGerber0`]
    - Smyth-Broby-Gerber 1 [`SmythBrobyGerber1`]
    - Smyth-Broby-Gerber 2 [`SmythBrobyGerber2`]
  - Standardised algorithms (Z-transforms the data beforehand)
    - Smyth-Broby 0 [`StandardisedSmythBroby0`]
    - Smyth-Broby 1 [`StandardisedSmythBroby1`]
    - Smyth-Broby 2 [`StandardisedSmythBroby2`]
    - Smyth-Broby-Gerber 0 [`StandardisedSmythBrobyGerber0`]
    - Smyth-Broby-Gerber 1 [`StandardisedSmythBrobyGerber1`]
    - Smyth-Broby-Gerber 2 [`StandardisedSmythBrobyGerber2`]
- Distance covariance with custom distance estimator via [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) [`DistanceCovariance`]
- Lower Tail Dependence covariance [`LowerTailDependenceCovariance`]
- Rank covariances
  - Kendall covariance [`KendallCovariance`]
  - Spearman covariance [`SpearmanCovariance`]
- Mutual information covariance with custom variance estimator and various binning algorithms [`MutualInfoCovariance`]
  - [`AstroPy`](https://docs.astropy.org/en/stable/stats/ref_api.html) provided bins
    - Knuth's optimal bin width [`Knuth`]
    - Freedman Diaconis bin width [`FreedmanDiaconis`]
    - Scott's bin width [`Scott`]
  - Hacine-Gharbi-Ravier bin width [`HacineGharbiRavier`]
  - Predefined number of bins
- Denoised covariance with custom covariance estimator [`DenoiseCovariance`]
- Detoned covariance with custom covariance estimator [`DetoneCovariance`]
- Custom processed covariance with custom covariance estimator [`ProcessedCovariance`]
- Implied volatility with custom covariance and matrix processing estimators, and implied volatility algorithms [`ImpliedVolatility`]-(TBA)
  - Premium [`ImpliedVolatilityPremium`]-(TBA)
  - Regression [`ImpliedVolatilityRegression`]-(TBA)
- Covariance with custom covariance estimator and matrix processing pipeline [`PortfolioOptimisersCovariance`]
- Correlation covariance [`CorrelationCovariance`]-(TBA)

#### [Coskewness](@id readme-coskewness)

Implements [`coskewness`].

- Coskewness and spectral decomposition of the negative coskewness with custom expected returns estimator and matrix processing pipeline [`Coskewness`]
  - Full coskewness [`Full`]
  - Semi (downside) coskewness [`Semi`]

#### [Cokurtosis](@id readme-cokurtosis)

Implements [`cokurtosis`].

- Cokurtosis with custom expected returns estimator and matrix processing pipeline [`Cokurtosis`]
  - Full cokurtosis [`Full`]
  - Semi (downside) cokurtosis [`Semi`]

### Distance matrices

Implements [`distance`] and [`cor_and_dist`].

- First order distance estimator with custom distance algorithm, and optional exponent [`Distance`]
- Second order distance estimator with custom pairwise distance algorithm from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl), custom distance algorithm, and optional exponent [`DistanceDistance`]

The distance estimators are used together with various distance matrix algorithms.

- Simple distance [`SimpleDistance`]
- Simple absolute distance [`SimpleAbsoluteDistance`]
- Logarithmic distance [`LogDistance`]
- Correlation distance [`CorrelationDistance`]
- Variation of Information distance with various binning algorithms [`VariationInfoDistance`]
  - [`AstroPy`](https://docs.astropy.org/en/stable/stats/ref_api.html) provided bins
    - Knuth's optimal bin width [`Knuth`]
    - Freedman Diaconis bin width [`FreedmanDiaconis`]
    - Scott's bin width [`Scott`]
  - Hacine-Gharbi-Ravier bin width [`HacineGharbiRavier`]
  - Predefined number of bins
- Canonical distance [`CanonicalDistance`]

### Phylogeny

`PortfolioOptimisers.jl` can make use of asset relationships to perform optimisations, define constraints, and compute relatedness characteristics of portfolios.

#### Clustering

Phylogeny constraints and clustering optimisations make use of clustering algorithms via [`ClustersEstimator`], [`Clusters`], and [`clusterise`]. Most clustering algorithms come from [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).

- Automatic choice of number of clusters via [`OptimalNumberClusters`] and [`VectorToScalarMeasure`]
  - Second order difference [`SecondOrderDifference`]
  - Silhouette scores [`SilhouetteScore`]
  - Predefined number of clusters.

##### Hierarchical

- Hierarchical clustering [`HClustAlgorithm`]
- Direct Bubble Hierarchical Trees [`DBHT`] and Local Global sparsification of the covariance matrix [`LoGo`], [`logo!`], and [`logo`]-(TBA)

##### Non-hierarchical

Non-hierarchical clustering algorithms are incompatible with hierarchical clustering optimisations, but they can be used for phylogeny constraints and [`NestedClustered`]-(TBA) optimisations.

- K-means clustering [`KMeansAlgorithm`]-(TBA)

#### Networks

##### Adjacency matrices

Adjacency matrices encode asset relationships either with clustering or graph theory via [`phylogeny_matrix`] and [`PhylogenyResult`].

- Network adjacency [`NetworkEstimator`] with custom tree algorithms, covariance, and distance estimators
  - Minimum spanning trees [`KruskalTree`], [`BoruvkaTree`], [`PrimTree`]
  - Triangulated Maximally Filtered Graph with various similarity matrix estimators
    - Maximum distance similarity [`MaximumDistanceSimilarity`]
    - Exponential similarity [`ExponentialSimilarity`]
    - General exponential similarity [`GeneralExponentialSimilarity`]
- Clustering adjacency [`ClustersEstimator`] and [`Clusters`]

##### Centrality and phylogeny measures

- Centrality estimator [`CentralityEstimator`] with custom adjacency matrix estimators (clustering and network) and centrality measures
  - Centrality measures
    - Betweenness [`BetweennessCentrality`]
    - Closeness [`ClosenessCentrality`]
    - Degree [`DegreeCentrality`]
    - Eigenvector [`EigenvectorCentrality`]
    - Katz [`KatzCentrality`]
    - Pagerank [`Pagerank`]
    - Radiality [`RadialityCentrality`]
    - Stress [`StressCentrality`]
- Centrality vector [`centrality_vector`]
- Average centrality [`average_centrality`]
- The asset phylogeny score [`asset_phylogeny`]

### Optimisation constraints

Non clustering optimisers support a wide range of constraints, while naive and clustering optimisers only support weight bounds. Furthermore, entropy pooling prior supports a variety of views constraints. It is therefore important to provide users with the ability to generate constraints manually and/or programmatically. We therefore provide a wide, robust, and extensible range of types such as [`AbstractEstimatorValueAlgorithm`] and [`UniformValues`], and functions that make this easy, fast, and safe.

Constraints can be defined via their estimators or directly by their result types. Some using estimators need to map key-value pairs to the asset universe, this is done by defining the assets and asset groups in [`AssetSets`]. Internally, `PortfolioOptimisers.jl` uses all the information and calls [`group_to_val!`], and [`replace_group_by_assets`] to produce the appropriate arrays.

- Equation parsing [`parse_equation`] and [`ParsingResult`].
- Linear constraints [`linear_constraints`], [`LinearConstraintEstimator`], [`PartialLinearConstraint`], and [`LinearConstraint`]
- Risk budgeting constraints [`risk_budget_constraints`], [`RiskBudgetEstimator`], and [`RiskBudget`]
- Phylogeny constraints [`phylogeny_constraints`], [`centrality_constraints`], [`SemiDefinitePhylogenyEstimator`], [`SemiDefinitePhylogeny`], [`IntegerPhylogenyEstimator`], [`IntegerPhylogeny`], [`CentralityConstraint`]
- Weight bounds constraints [`weight_bounds_constraints`], [`WeightBoundsEstimator`], [`WeightBounds`]
- Asset set matrices [`asset_sets_matrix`] and [`AssetSetsMatrixEstimator`]
- Threshold constraints [`threshold_constraints`], [`ThresholdEstimator`], and [`Threshold`]

### Prior statistics

Many optimisations and constraints use prior statistics computed via [`prior`].

- Low order prior [`LowOrderPrior`]
  - Empirical [`EmpiricalPrior`]
  - Factor model [`FactorPrior`]
  - Black-Litterman
    - Vanilla [`BlackLittermanPrior`]
    - Bayesian [`BayesianBlackLittermanPrior`]
    - Factor model [`FactorBlackLittermanPrior`]
    - Augmented [`AugmentedBlackLittermanPrior`]
  - Entropy pooling [`EntropyPoolingPrior`]
  - Opinion pooling [`OpinionPoolingPrior`]
- High order prior [`HighOrderPrior`]
  - High order [`HighOrderPriorEstimator`]
  - High order factor model [`HighOrderFactorPriorEstimator`]-(TBA)

### Uncertainty sets

In order to make optimisations more robust to noise and measurement error, it is possible to define uncertainty sets on the expected returns and covariance. These can be used in optimisations which use either of these two quantities. These are implemented via [`ucs`], [`mu_ucs`], and [`sigma_ucs`].

`PortfolioOptimisers.jl` implements two types of uncertainty sets.

- [`BoxUncertaintySet`] and [`BoxUncertaintySetAlgorithm`]
- [`EllipsoidalUncertaintySet`] and [`EllipsoidalUncertaintySetAlgorithm`] with various algorithms for computing the scaling parameter via [`k_ucs`]
  - [`NormalKUncertaintyAlgorithm`]
  - [`GeneralKUncertaintyAlgorithm`]
  - [`ChiSqKUncertaintyAlgorithm`]
  - Predefined scaling parameter

It also implements various estimators for the uncertainty sets, the following two can generate box and ellipsoidal sets.

- Normally distributed returns [`NormalUncertaintySet`]
- Bootstrapping via Autoregressive Conditional Heteroscedasticity [`ARCHUncertaintySet`] via [`arch`](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html)
  - Circular [`CircularBootstrap`]
  - Moving [`MovingBootstrap`]
  - Stationary [`StationaryBootstrap`]

The following estimator can only generate box sets.

- [`DeltaUncertaintySet`]

### [Turnover](@id readme-turnover)

The turnover is defined as the element-wise absolute difference between the vector of current weights and a vector of benchmark weights. It can be used as a constraint, method for fee calculation, and risk measure. These are all implemented using [`turnover_constraints`], [`TurnoverEstimator`], and [`Turnover`].

### Fees

Fees are a non-negligible aspect of active investing. As such `PortfolioOptimiser.jl` has the ability to account for them in all optimisations but the naive ones. They can also be used to adjust expected returns calculations via [`calc_fees`] and [`calc_asset_fees`].

- Fees [`FeesEstimator`] and [`Fees`]
  - Proportional long
  - Proportional short
  - Fixed long
  - Fixed short
  - Turnover

### Portfolio returns and drawdowns

Various risk measures and analyses require the computation of simple and cumulative portfolio returns and drawdowns both in aggregate and per-asset. These are computed by [`calc_net_returns`], [`calc_net_asset_returns`], [`cumulative_returns`], [`drawdowns`].

### [Tracking](@id readme-tracking)

It is often useful to create portfolios that track the performance of an index, indicator, or another portfolio.

- Tracking error [`tracking_benchmark`], [`TrackingError`]
  - Returns tracking [`ReturnsTracking`]
  - Weights tracking [`WeightsTracking`]

The error can be computed using different algorithms using [`norm_tracking`].

- L1-norm [`NOCTracking`]
- L2-norm [`SOCTracking`]
- L2-norm squared [`SquaredSOCTracking`]

It is also possible to track the error in with risk measures [`RiskTrackingError`]-(TBA) using [`WeightsTracking`], which allows for two approaches.

- Dependent variable tracking [`DependentVariableTracking`]
- Independent variable tracking [`IndependentVariableTracking`]

### Risk measures

`PortfolioOptimisers.jl` provides a wide range of risk measures. These are broadly categorised into two types based on the type of optimisations that support them.

#### Risk measures for traditional optimisation

These are all subtypes of [`RiskMeasure`], and are supported by all optimisation estimators.

- Variance [`Variance`]
  - Traditional optimisations also support:
    - Risk contribution
    - Formulations
      - Quadratic risk expression [`QuadRiskExpr`]
      - Squared second order cone [`SquaredSOCRiskExpr`]
- Standard deviation [`StandardDeviation`]
- Uncertainty set variance [`UncertaintySetVariance`] (same as variance when used in non-traditional optimisation)
- Low order moments [`LowOrderMoment`]
  - First lower moment [`FirstLowerMoment`]
  - Mean absolute deviation [`MeanAbsoluteDeviation`]
  - Second moment [`SecondMoment`]
    - Second squared moments
      - Scenario variance [`Full`]
      - Scenario semi-variance [`Semi`]
      - Traditional optimisation formulations
        - Quadratic risk expression [`QuadRiskExpr`]
        - Squared second order cone [`SquaredSOCRiskExpr`]
        - Rotated second order cone [`RSOCRiskExpr`]
    - Second moments [`SOCRiskExpr`]
      - Scenario standard deviation [`Full`]
      - Scenario semi-standard deviation [`Semi`]
- Kurtosis [`Kurtosis`]
  - Actual kurtosis
    - Full and semi-kurtosis are supported in traditional optimisers via the `kt` field. Risk calculation uses
      - Kurtosis [`Full`]
      - Semi-kurtosis [`Semi`]
    - Traditional optimisation formulations
      - Quadratic risk expression [`QuadRiskExpr`]
      - Squared second order cone [`SquaredSOCRiskExpr`]
      - Rotated second order cone [`RSOCRiskExpr`]
  - Square root kurtosis [`SOCRiskExpr`]
    - Full [`Full`]
    - Semi [`Semi`]
- Negative skewness [`NegativeSkewness`]-(TBA)
  - Squared negative skewness
    - Full and semi-skewness are supported in traditional optimisers via the `sk` and `V` fields. Risk calculation uses
      - Negative skewness [`Full`]
      - Negative semi-skewness [`Semi`]
    - Traditional optimisation formulations
      - Quadratic risk expression [`QuadRiskExpr`]
      - Squared second order cone [`SquaredSOCRiskExpr`]
    - Square root negative skewness [`SOCRiskExpr`]
- Value at Risk [`ValueatRisk`]-(TBA)
  - Traditional optimisation formulations
    - Exact MIP formulation [`MIPValueatRisk`]-(TBA)
    - Approximate distribution based [`DistributionValueatRisk`]-(TBA)
- Value at Risk Range [`ValueatRiskRange`]-(TBA)
  - Traditional optimisation formulations
    - Exact MIP formulation [`MIPValueatRisk`]-(TBA)
    - Approximate distribution based [`DistributionValueatRisk`]-(TBA)
- Drawdown at Risk [`DrawdownatRisk`]-(TBA)
- Conditional Value at Risk [`ConditionalValueatRisk`]-(TBA)
- Distributionally Robust Conditional Value at Risk [`DistributionallyRobustConditionalValueatRisk`]-(TBA) (same as conditional value at risk when used in non-traditional optimisation)
- Conditional Value at Risk Range [`ConditionalValueatRiskRange`]-(TBA)
- Distributionally Robust Conditional Value at Risk Range [`DistributionallyRobustConditionalValueatRiskRange`]-(TBA) (same as conditional value at risk range when used in non-traditional optimisation)
- Conditional Drawdown at Risk [`ConditionalDrawdownatRisk`]-(TBA)
- Distributionally Robust Conditional Drawdown at Risk [`DistributionallyRobustConditionalDrawdownatRisk`]-(TBA)(same as conditional drawdown at risk when used in non-traditional optimisation)
- Entropic Value at Risk [`EntropicValueatRisk`]-(TBA)
- Entropic Value at Risk Range [`EntropicValueatRiskRange`]-(TBA)
- Entropic Drawdown at Risk [`EntropicDrawdownatRisk`]-(TBA)
- Relativistic Value at Risk [`RelativisticValueatRisk`]-(TBA)
- Relativistic Value at Risk Range [`RelativisticValueatRiskRange`]-(TBA)
- Relativistic Drawdown at Risk [`RelativisticDrawdownatRisk`]-(TBA)
- Ordered Weights Array
  - Risk measures
    - Ordered Weights Array risk measure [`OrderedWeightsArray`]-(TBA)
    - Ordered Weights Array range risk measure [`OrderedWeightsArrayRange`]-(TBA)
  - Traditional optimisation formulations
    - Exact [`ExactOrderedWeightsArray`]-(TBA)
    - Approximate [`ApproxOrderedWeightsArray`]-(TBA)
  - Array functions
    - Gini Mean Difference [`owa_gmd`]
    - Worst Realisation [`owa_wr`]
    - Range [`owa_rg`]
    - Conditional Value at Risk [`owa_cvar`]
    - Weighted Conditional Value at Risk [`owa_wcvar`]
    - Conditional Value at Risk Range [`owa_cvarrg`]
    - Weighted Conditional Value at Risk Range [`owa_wcvarrg`]
    - Tail Gini [`owa_tg`]
    - Tail Gini Range [`owa_tgrg`]
    - Linear moments (L-moments)
      - Linear Moment [`owa_l_moment`]
      - Linear Moment Convex Risk Measure [`owa_l_moment_crm`]
        - L-moment combination formulations
          - Maximum Entropy [`MaximumEntropy`]-(TBA)
            - Entropy formulations
              - Exponential Cone Entropy [`ExponentialConeEntropy`]-(TBA)
              - Relative Entropy [`RelativeEntropy`]-(TBA)
          - Minimum Squared Distance [`MinimumSquaredDistance`]-(TBA)
          - Minimum Sum Squares [`MinimumSumSquares`]-(TBA)
- Average Drawdown [`AverageDrawdown`]-(TBA)
- Ulcer Index [`UlcerIndex`]-(TBA)
- Maximum Drawdown [`MaximumDrawdown`]-(TBA)
- Brownian Distance Variance [`BrownianDistanceVariance`]-(TBA)
  - Traditional optimisation formulations
    - Distance matrix constraint formulations
      - Norm one cone Brownian distance variance [`NormOneConeBrownianDistanceVariance`]-(TBA)
      - Inequality Brownian distance variance [`IneqBrownianDistanceVariance`]-(TBA)
    - Risk formulation
      - Quadratic risk expression [`QuadRiskExpr`]
      - Rotated second order cone [`RSOCRiskExpr`]
- Worst Realisation [`WorstRealisation`]-(TBA)
- Range [`Range`]-(TBA)
- Turnover Risk Measure [`TurnoverRiskMeasure`]-(TBA)
- Tracking Risk Measure [`TrackingRiskMeasure`]-(TBA)
  - Formulations
    - L1-norm [`NOCTracking`]
    - L2-norm [`SOCTracking`]
    - L2-norm squared [`SquaredSOCTracking`]
- Risk Tracking Risk Measure
  - Formulations
    - Dependent variable tracking [`DependentVariableTracking`]
    - Independent variable tracking [`IndependentVariableTracking`]
- Power Norm Value at Risk [`PowerNormValueatRisk`]-(TBA)
- Power Norm Value at Risk Range [`PowerNormValueatRiskRange`]-(TBA)
- Power Norm Drawdown at Risk [`PowerNormDrawdownatRisk`]-(TBA)

#### Risk measures for hierarchical optimisation

These are all subtypes of [`HierarchicalRiskMeasure`], and are only supported by hierarchical optimisation estimators.

- High order moment [`HighOrderMoment`]
  - Unstandardised third lower moment [`ThirdLowerMoment`]
  - Standardised third lower moment [`StandardisedHighOrderMoment`] and [`ThirdLowerMoment`]
  - Unstandardised fourth moment [`FourthMoment`]
    - Full [`Full`]
    - Semi [`Semi`]
  - Standardised fourth moment [`StandardisedHighOrderMoment`] and [`FourthMoment`]
    - Full [`Full`]
    - Semi [`Semi`]
- Relative Drawdown at Risk [`RelativeDrawdownatRisk`]-(TBA)
- Relative Conditional Drawdown at Risk [`RelativeConditionalDrawdownatRisk`]-(TBA)
- Relative Entropic Drawdown at Risk [`RelativeEntropicDrawdownatRisk`]-(TBA)
- Relative Relativistic Drawdown at Risk [`RelativeRelativisticDrawdownatRisk`]-(TBA)
- Relative Average Drawdown [`RelativeAverageDrawdown`]-(TBA)
- Relative Ulcer Index [`RelativeUlcerIndex`]-(TBA)
- Relative Maximum Drawdown [`RelativeMaximumDrawdown`]-(TBA)
- Relative Power Norm Drawdown at Risk [`RelativePowerNormDrawdownatRisk`]-(TBA)
- Risk Ratio Risk Measure [`RiskRatioRiskMeasure`]-(TBA)
- Equal Risk Measure [`EqualRiskMeasure`]-(TBA)
- Median Absolute Deviation [`MedianAbsoluteDeviation`]-(TBA)

#### Non-optimisation risk measures

These risk measures are unsuitable for optimisation because they can return negative values. However, they can be used for performance metrics.

- Mean Return [`MeanReturn`]-(TBA)
- Third Central Moment [`ThirdCentralMoment`]-@(ref)
- Skewness [`Skewness`]-(TBA)
- Return Risk Measure [`ReturnRiskMeasure`]
- Return Risk Ratio Risk Measure [`ReturnRiskRatioRiskMeasure`]

### Performance metrics

- Expected risk [`expected_risk`]-(TBA)
- Number of effective assets [`number_effective_assets`]-(TBA)
- Risk contribution
  - Asset risk contribution [`risk_contribution`]-(TBA)
  - Factor risk contribution [`factor_risk_contribution`]-(TBA)
- Expected return [`expected_return`]
  - Arithmetic [`ArithmeticReturn`]-(TBA)
  - Logarithmic [`LogarithmicReturn`]-(TBA)
- Expected risk-adjusted return ratio [`expected_ratio`] and [`expected_risk_ret_ratio`]
- Expected risk-adjusted ratio information criterion [`expected_sric`] and [`expected_risk_ret_sric`]
- Brinson performance attribution [`brinson_attribution`]

### Portfolio optimisation

Optimisations are implemented via [`optimise`]-(TBA). Optimisations consume an estimator and return a result.

#### Naive

These return a [`NaiveOptimisationResult`]-(TBA).

- Inverse Volatility [`InverseVolatility`]-(TBA)
- Equal Weighted [`EqualWeighted`]-(TBA)
- Random (Dirichlet) [`RandomWeighted`]-(TBA)

##### Naive optimisation features

- Weight bounds [`WeightBoundsEstimator`], [`UniformValues`], and [`WeightBounds`]
- Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(TBA)
  - JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(TBA)
    - Error formulations
      - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(TBA)
      - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(TBA)
      - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(TBA)
      - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(TBA)

#### Traditional

These optimisations are implemented as `JuMP` problems and make use of [`JuMPOptimiser`]-(TBA), which encodes all supported constraints.

##### Objective function optimisations

These optimisations support a variety of objective functions.

- Objective functions
  - Minimum risk [`MinimumRisk`]-(TBA)
  - Maximum utility [`MaximumUtility`]-(TBA)
  - Maximum return over risk ratio [`MaximumRatio`]-(TBA)
  - Maximum return [`MaximumReturn`]-(TBA)
- Exclusive to [`MeanRisk`]-(TBA) and [`NearOptimalCentering`]-(TBA)
  - N-dimensional Pareto fronts [`Frontier`]
    - Return based
    - Risk based
- Optimisation estimators
  - Mean-Risk [`MeanRisk`]-(TBA) returns a [`MeanRiskResult`]-(TBA)
  - Near Optimal Centering [`NearOptimalCentering`]-(TBA) returns a [`NearOptimalCenteringResult`]-(TBA)
  - Factor Risk Contribution [`FactorRiskContribution`]-(TBA) returns a [`FactorRiskContributionResult`]-(TBA)

##### Risk budgeting optimisations

These optimisations attempt to achieve weight values according to a risk budget vector. This vector can be provided on a per asset or per factor basis.

- Budget targets
  - Asset risk budgeting [`AssetRiskBudgeting`]-(TBA)
  - Factor risk budgeting [`FactorRiskBudgeting`]-(TBA)
- Optimisation estimators
  - Risk Budgeting [`RiskBudgeting`]-(TBA) returns a [`RiskBudgetingResult`]-(TBA)
  - Relaxed Risk Budgeting [`RelaxedRiskBudgeting`]-(TBA) returns a [`RiskBudgetingResult`]-(TBA)
    - Relaxed risk budgeting types
      - Basic [`BasicRelaxedRiskBudgeting`]-(TBA)
      - Regularised [`RegularisedRelaxedRiskBudgeting`]-(TBA)
      - Regularised and penalised [`RegularisedPenalisedRelaxedRiskBudgeting`]-(TBA)

##### Traditional optimisation features

- Custom objective penalty [`CustomJuMPObjective`]-(TBA)
- Weight bounds [`WeightBoundsEstimator`], [`UniformValues`], and [`WeightBounds`]
- Budget
  - Directionality
    - Long
    - Short
  - Type
    - Exact
    - Range [`BudgetRange`]-(TBA)
- Threshold [`ThresholdEstimator`] and [`Threshold`]
  - Directionality
    - Long
    - Short
  - Type
    - Asset
    - Set [`AssetSetsMatrixEstimator`]
- Linear constraints [`LinearConstraintEstimator`] and [`LinearConstraint`]
- Centralit(y/ies) [`CentralityEstimator`]
- Cardinality
  - Asset
  - Asset group(s) [`LinearConstraintEstimator`] and [`LinearConstraint`]
  - Set(s)
  - Set group(s) [`LinearConstraintEstimator`] and [`LinearConstraint`]
- Turnover(s) [`TurnoverEstimator`] and [`Turnover`]
- Fees [`FeesEstimator`] and [`Fees`]
- Tracking error(s) [`TrackingError`]
- Phylogen(y/ies) [`IntegerPhylogenyEstimator`] and [`SemiDefinitePhylogenyEstimator`]
- Portfolio returns
  - Arithmetic returns [`ArithmeticReturn`]-(TBA)
    - Uncertainty set [`BoxUncertaintySet`], [`BoxUncertaintySetAlgorithm`], [`EllipsoidalUncertaintySet`], and [`EllipsoidalUncertaintySetAlgorithm`]
    - Custom expected returns vector
  - Logarithmic returns [`LogarithmicReturn`]-(TBA)
- Risk vector scalarisation
  - Weighted sum [`SumScalariser`]
  - Maximum value [`MaxScalariser`]
  - Log-sum-exp [`LogSumExpScalariser`]
- Custom constraint
- Number of effective assets
- Regularisation penalty
  - L1
  - L2

#### [Clustering](@id readme-clustering-opt)

Clustering optimisations make use of asset relationships to either minimise the risk exposure by breaking the asset universe into subsets which are hierarchically or individually optimised.

##### Hierarchical clustering optimisations

These optimisations minimise risk by hierarchically splitting the asset universe into subsets, computing the risk of each subset, and combining them according to their hierarchy.

- Hierarchical Risk Parity [`HierarchicalRiskParity`]-(TBA) returns a [`HierarchicalResult`]-(TBA)
- Hierarchical Equal Risk Contribution [`HierarchicalEqualRiskContribution`]-(TBA) returns a [`HierarchicalResult`]-(TBA)

###### Hierarchical clustering optimisation features

- Weight bounds [`WeightBoundsEstimator`], [`UniformValues`], and [`WeightBounds`]
- Fees [`FeesEstimator`] and [`Fees`]
- Risk vector scalarisation
  - Weighted sum [`SumScalariser`]
  - Maximum value [`MaxScalariser`]
  - Log-sum-exp [`LogSumExpScalariser`]
- Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(TBA)
  - JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(TBA)
    - Error formulations
      - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(TBA)
      - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(TBA)
      - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(TBA)
      - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(TBA)

##### Schur complementary optimisation

Schur complementary hierarchical risk parity provides a bridge between mean variance optimisation and hierarchical risk parity by using an interpolation parameter. It converges to hierarchical risk parity, and approximates mean variance by adjusting this parameter. It uses the Schur complement to adjust the weights of a portfolio according to how much more useful information is gained by assigning more weight to a group of assets.

- Schur Complementary Hierarchical Risk Parity [`SchurComplementHierarchicalRiskParity`]-(TBA) returns a [`SchurComplementHierarchicalRiskParityResult`]-(TBA)

###### Schur complementary optimisation features

- Weight bounds [`WeightBoundsEstimator`], [`UniformValues`], and [`WeightBounds`]
- Fees [`FeesEstimator`] and [`Fees`]
- Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(TBA)
  - JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(TBA)
    - Error formulations
      - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(TBA)
      - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(TBA)
      - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(TBA)
      - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(TBA)

##### Clustering optimisation

Nested clustered optimisation breaks the asset universe into smaller subsets and treats every subset as an individual portfolio. Then it creates a synthetic asset out of each portfolio, optimises the portfolio of synthetic assets. The final weights are the inner product between the individual portfolio weights and outer portfolio.

- Nested Clustered [`NestedClustered`]-(TBA) returns a [`NestedClusteredResult`]-(TBA)

##### Clustering optimisation features

- Any features supported by the inner and outer estimators.
- Weight bounds [`WeightBoundsEstimator`], [`UniformValues`], and [`WeightBounds`]
- Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(TBA)
  - JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(TBA)
    - Error formulations
      - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(TBA)
      - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(TBA)
      - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(TBA)
      - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(TBA)
- Cross validation predictor for the outer estimator

#### Ensemble optimisation

These work similar to the Nested Clustered estimator, only instead of breaking the asset universe into subsets, a list of inner estimators is provided, all of which are optimised, and each result is treated as a synthetic asset from which a synthetic portfolio is created and optimised according to an outer estimator. The final weights are the inner product between the individual portfolio weights and outer portfolio.

- Stacking [`Stacking`]-(TBA) returns a [`StackingResult`]-(TBA)

##### Ensemble optimisation features

- Any features supported by the inner and outer estimators.
- Weight bounds [`WeightBoundsEstimator`], [`UniformValues`], and [`WeightBounds`]
- Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(TBA)
  - JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(TBA)
    - Error formulations
      - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(TBA)
      - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(TBA)
      - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(TBA)
      - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(TBA)
- Cross validation predictor for the outer estimator

#### Finite allocation optimisation

Unlike all other estimators, finite allocation does not yield an "optimal" value, but rather the optimal attainable solution based on a finite amount of capital. They use the result of other estimations, the latest prices, and a cash amount.

- Discrete (MIP) [`DiscreteAllocation`]-(TBA)
  - Weight finalisers
    - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(TBA)
    - JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(TBA)
      - Error formulations
        - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(TBA)
        - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(TBA)
        - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(TBA)
        - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(TBA)
- Greedy [`GreedyAllocation`]

### Plotting

Visualising the results is quite a useful way of summarising the portfolio characteristics or evolution. To this extent we provide a few plotting functions with more to come.

- Simple or compound cumulative returns.
  - Portfolio [`plot_ptf_cumulative_returns`]-(TBA).
  - Assets [`plot_asset_cumulative_returns`]-(TBA).
- Portfolio composition.
  - Single portfolio [`plot_composition`]-(TBA).
  - Multi portfolio.
    - Stacked bar [`plot_stacked_bar_composition`]-(TBA).
    - Stacked area [`plot_stacked_area_composition`]-(TBA).
- Risk contribution.
  - Asset risk contribution [`plot_risk_contribution`]-(TBA).
  - Factor risk contribution [`plot_factor_risk_contribution`]-(TBA).
- Asset dendrogram [`plot_dendrogram`]-(TBA).
- Asset clusters + optional dendrogram [`plot_clusters`]-(TBA).
- Simple or compound drawdowns [`plot_drawdowns`]-(TBA).
- Portfolio returns histogram + density [`plot_histogram`]-(TBA).
- 2/3D risk measure scatter plots [`plot_measures`]-(TBA).
