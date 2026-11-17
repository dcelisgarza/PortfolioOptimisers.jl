#=
```@meta
Description = "The maximum diversification portfolio in PortfolioOptimisers.jl, as a MaximumRatio of the asset volatilities to the portfolio standard deviation."
```

# [Maximum diversification](@id example-maximum-diversification)

The diversification ratio of a portfolio is the weighted average of the volatilities of its
assets, divided by the volatility of the portfolio.

```math
\mathrm{DR}(\boldsymbol{w}) = \frac{\boldsymbol{w}^\intercal \boldsymbol{\sigma}}{\sqrt{\boldsymbol{w}^\intercal \mathbf{\Sigma} \boldsymbol{w}}}\,,
```

where ``\boldsymbol{\sigma}`` is the vector of the volatilities of the assets and
``\mathbf{\Sigma}`` is their covariance matrix. The ratio is one for a portfolio of one asset,
and it grows as the correlations between the assets cut the risk of the portfolio. The maximum
diversification portfolio maximises it.

The library has no optimiser of that name, and it needs none. The ratio is a
[`MaximumRatio`](@ref) problem of [`MeanRisk`](@ref):

  - the numerator is the expected return of the portfolio, when the expected return of each
    asset is its volatility. [`StandardDeviationExpectedReturns`](@ref) gives this vector to
    the prior;
  - the denominator is the [`StandardDeviation`](@ref) risk measure;
  - the risk-free rate is zero, the default of [`MaximumRatio`](@ref).

This page builds the portfolio, checks that it has the largest diversification ratio, and
compares it with the [`InverseVolatility`](@ref) and the minimum-variance portfolios.

!!! tip "When to reach for this"
    Reach for the maximum diversification portfolio when you do not trust the estimates of the
    expected returns, but you want more than the minimum-variance portfolio. It uses only the
    covariance matrix, as the minimum-variance portfolio does, but it does not prefer the assets
    of low volatility. It prefers the assets of low correlation to the others.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS,
      Pajarito, JuMP, StatsPlots, GraphRecipes, LinearAlgebra

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, AbstractFloat) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data and the prior

We load the last 253 days of the S&P 500 prices that the other optimiser examples use, and
compute their returns.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
The prior takes [`StandardDeviationExpectedReturns`](@ref) as its expected returns estimator,
`me`. The mean vector of the prior is then the vector of the volatilities. Its covariance
estimator has the same default as the covariance estimator of the prior, so
``\boldsymbol{\sigma}`` is the square root of the diagonal of ``\mathbf{\Sigma}``.
=#

pr = prior(EmpiricalPrior(; me = StandardDeviationExpectedReturns()), rd)
isapprox(pr.mu, sqrt.(diag(pr.sigma)))

#=
## 2. The maximum diversification portfolio

The return measure of [`MeanRisk`](@ref) is [`ArithmeticReturn`](@ref) by default, and it
reads the mean vector of the prior. So the expected return of the portfolio is
``\boldsymbol{w}^\intercal \boldsymbol{\sigma}``, and [`MaximumRatio`](@ref) with `rf = 0`
maximises the diversification ratio.

The risk-free rate must be zero. A rate `rf` other than zero maximises
``(\boldsymbol{w}^\intercal \boldsymbol{\sigma} - r_f) / \sqrt{\boldsymbol{w}^\intercal \mathbf{\Sigma} \boldsymbol{w}}``,
which is a different problem with a different portfolio.
=#

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
opt = JuMPOptimiser(; pe = pr, slv = slv)
r = StandardDeviation()

res_md = optimise(MeanRisk(; r = r, obj = MaximumRatio(), opt = opt))

#=
## 3. Three benchmarks

The minimum-variance portfolio uses the same risk measure and the same optimiser, with the
[`MinimumRisk`](@ref) objective. The [`InverseVolatility`](@ref) portfolio gives each asset a
weight in proportion to the inverse of its volatility. It needs no solver. The
[`EqualWeighted`](@ref) portfolio gives each asset the same weight.
=#

res_mv = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
res_iv = optimise(InverseVolatility(; pe = pr))
res_ew = optimise(EqualWeighted(), rd)

pretty_table(DataFrame(; :assets => rd.nx, Symbol("max diversification") => res_md.w,
                       Symbol("min variance") => res_mv.w,
                       Symbol("inverse volatility") => res_iv.w); formatters = [resfmt])

#=
## 4. The diversification ratio of each portfolio

[`expected_risk_ret_ratio`](@ref) returns the risk, the return and the ratio of a portfolio. With
the prior of section 1, the return is ``\boldsymbol{w}^\intercal \boldsymbol{\sigma}``, so the
ratio is the diversification ratio. The default risk-free rate of the function is zero.
=#

dr(w) = expected_risk_ret_ratio(r, ArithmeticReturn(), w, pr)[3]

portfolios = ["max diversification" => res_md, "min variance" => res_mv,
              "inverse volatility" => res_iv, "equal weighted" => res_ew]
pretty_table(DataFrame(; :portfolio => first.(portfolios),
                       Symbol("diversification ratio") =>
                           [dr(p.w) for p in last.(portfolios)],
                       :volatility =>
                           [expected_risk(r, p.w, pr) for p in last.(portfolios)],
                       Symbol("assets held") =>
                           [count(>(1e-6), p.w) for p in last.(portfolios)]);
             formatters = [(v, i, j) -> j == 2 ? round(v; digits = 4) : resfmt(v, i, j)])

#=
The maximum diversification portfolio has the largest diversification ratio of the four.
=#

all(dr(res_md.w) > dr(res.w) for res in (res_iv, res_mv, res_ew))

#=
The minimum-variance portfolio has the lowest volatility, and the maximum diversification
portfolio does not. The two portfolios answer different questions: the first asks for the
least risk, and the second asks for the most risk that the correlations remove.

## 5. The minimum variance of the correlation matrix

Write the weights as ``w_i = y_i / \sigma_i``, up to a positive scale. The diversification ratio
is then ``\sum_i y_i / \sqrt{\boldsymbol{y}^\intercal \mathbf{C} \boldsymbol{y}}``, where
``\mathbf{C}`` is the correlation matrix. With the budget ``\sum_i y_i = 1``, to maximise the
ratio is to minimise the variance ``\boldsymbol{y}^\intercal \mathbf{C} \boldsymbol{y}``. So
the maximum diversification portfolio is the minimum-variance portfolio of the correlation
matrix, divided by the volatilities and scaled to the budget.

[`CorrelationCovariance`](@ref) returns the correlation matrix of the estimator that it wraps,
so a prior with it as `ce` has ``\mathbf{C}`` in the place of ``\mathbf{\Sigma}``. We solve the
minimum-variance problem on that prior, and compare the result with the portfolio of section 2.
=#

pr_cor = prior(EmpiricalPrior(;
                              ce = CorrelationCovariance(;
                                                         ce = PortfolioOptimisersCovariance())),
               rd)
res_cor = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr_cor, slv = slv)))
w_cor = res_cor.w ./ pr.mu
w_cor ./= sum(w_cor)
isapprox(w_cor, res_md.w; atol = 1e-4)

#=
The two solves give the same weights, to the tolerance of the solver.

## 6. A cap on the number of assets

The maximum diversification portfolio of section 2 holds most of the assets. The `card` keyword
of [`JuMPOptimiser`](@ref) caps the number of assets that hold a weight. It adds binary
variables, so it needs a mixed-integer solver. We use Pajarito, which uses HiGHS for the integer
part and Clarabel for the conic part, as the
[cardinality and threshold](@ref example-cardinality-and-threshold) example does.
=#

mip_slv = Solver(; name = :mip1,
                 solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                    "oa_solver" =>
                                                        optimizer_with_attributes(HiGHS.Optimizer,
                                                                                  JuMP.MOI.Silent() =>
                                                                                      true),
                                                    "conic_solver" =>
                                                        optimizer_with_attributes(Clarabel.Optimizer,
                                                                                  "verbose" =>
                                                                                      false)),
                 check_sol = (; allow_local = true, allow_almost = true))
k = 5
res_card = optimise(MeanRisk(; r = r, obj = MaximumRatio(),
                             opt = JuMPOptimiser(; pe = pr, slv = mip_slv, card = k)))

#=
An [`L1UncertaintySet`](@ref) gives a portfolio of `k` assets with no binary variables. The
[ℓ1 uncertainty sets](@ref example-l1-uncertainty-sets-the-quintile-and-1-n-portfolios) example
shows the model. [`MaximumReturn`](@ref) with [`NoRisk`](@ref) maximises the worst-case return
over the set, and the result holds the `k` assets with the highest characteristic, with equal
weights. It is a linear program, so HiGHS solves it alone.

The set must be centred on the return term of this page, the volatilities. So its prior takes
[`StandardDeviationExpectedReturns`](@ref), as the prior of section 1 does.
[`ActiveAssetsUncertaintyAlgorithm`](@ref) finds the radius that makes `k` assets active.
=#

ue = CharacteristicUncertaintySet(;
                                  pe = EmpiricalPrior(;
                                                      me = StandardDeviationExpectedReturns()),
                                  alg = L1UncertaintySetAlgorithm(;
                                                                  method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                            active = CountOf(k))))
lp_slv = Solver(; name = :highs,
                solver = optimizer_with_attributes(HiGHS.Optimizer,
                                                   JuMP.MOI.Silent() => true),
                check_sol = (; allow_local = true, allow_almost = true))
res_l1 = optimise(MeanRisk(; r = NoRisk(), obj = MaximumReturn(),
                           opt = JuMPOptimiser(; pe = pr, slv = lp_slv,
                                               ret = ArithmeticReturn(;
                                                                      ucs = mu_ucs(ue, rd)))))

pretty_table(DataFrame(; :portfolio => ["card = $k", "ℓ1 set, $k active", "no cap"],
                       Symbol("assets held") =>
                           [count(>(1e-6), p.w) for p in (res_card, res_l1, res_md)],
                       Symbol("diversification ratio") =>
                           [dr(p.w) for p in (res_card, res_l1, res_md)]);
             formatters = [(v, i, j) -> j == 3 ? round(v; digits = 4) : v])

#=
Both forms hold `k` assets.
=#

all(count(>(1e-6), p.w) == k for p in (res_card, res_l1))

#=
The two forms do not select the same assets. The ℓ1 set ranks the assets on its characteristic,
so it holds the `k` assets of the highest volatility. That is not the selection that maximises
the diversification ratio, so its ratio is lower than the ratio of the `card` portfolio. The ℓ1
set replaces the binary variables when the ranking on the characteristic is the selection that
you want.

The ℓ1 set can also enter a [`MinimumRisk`](@ref) problem, through a lower bound on the
worst-case return. The bound must be the largest worst-case return that the constraints let the
portfolio reach. A bound slot holds a value that you give before the solve, and you know that
value only after you solve the problem above. A larger bound, such as the largest volatility,
makes the problem infeasible. So use the form above.

## 7. The compositions

The stacked bars show the weights of the five portfolios.
=#

plot_stacked_bar_composition([res_md, res_mv, res_iv, res_card, res_l1], rd;
                             xticks = (1:5,
                                       ["max diversification", "min variance",
                                        "inverse volatility", "card = $k", "ℓ1 set"]))
